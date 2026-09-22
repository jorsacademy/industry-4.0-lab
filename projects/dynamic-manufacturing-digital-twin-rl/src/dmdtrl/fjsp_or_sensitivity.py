from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dmdtrl.fjsp_evaluate import instance_fingerprint
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_optimization import FJSPCPSATConfig, FJSPRollingHorizonCPSAT
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator
from dmdtrl.statistics import bootstrap_mean_ci, paired_estimate

DEFAULT_HORIZONS = (4, 8, 12)
DEFAULT_SOLVER_BUDGETS = (0.02, 0.05, 0.10)


def variant_name(horizon: int, solver_seconds: float) -> str:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if solver_seconds <= 0.0:
        raise ValueError("solver_seconds must be positive")
    budget_ms = int(round(1_000.0 * solver_seconds))
    return f"FJSP_CPSAT_H{horizon}_B{budget_ms}MS"


def sensitivity_grid(
    horizons: Sequence[int], solver_budgets: Sequence[float]
) -> list[tuple[int, float]]:
    if not horizons or not solver_budgets:
        raise ValueError("horizon and solver-budget grids must be non-empty")
    grid = [(int(h), float(b)) for h in horizons for b in solver_budgets]
    if any(h <= 0 for h, _ in grid) or any(b <= 0.0 for _, b in grid):
        raise ValueError("all horizons and solver budgets must be positive")
    names = [variant_name(h, b) for h, b in grid]
    if len(names) != len(set(names)):
        raise ValueError("sensitivity grid contains duplicate configurations")
    if len(grid) < 2:
        raise ValueError("sensitivity grid must contain at least two configurations")
    return grid


def run_variant(
    *,
    seed: int,
    generator_config: FJSPGeneratorConfig,
    setup_time: float,
    horizon: int,
    solver_seconds: float,
) -> dict[str, float | int | str]:
    instance = generate_fjsp_instance(np.random.default_rng(seed), generator_config)
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=setup_time)
    controller = FJSPRollingHorizonCPSAT(
        FJSPCPSATConfig(job_horizon=horizon, solver_seconds=solver_seconds)
    )
    while not simulator.terminated:
        decision = controller.choose(simulator)
        simulator.step(decision.action)
    stats = controller.stats()
    return {
        "seed": seed,
        "seed_regime": "validation" if 41000 <= seed < 42000 else "development_or_final",
        "instance_sha256": instance_fingerprint(instance),
        "policy": variant_name(horizon, solver_seconds),
        "cpsat_horizon": horizon,
        "solver_budget_ms": 1_000.0 * solver_seconds,
        **simulator.metrics(),
        "mean_decision_time_ms": stats["mean_solve_time_ms"],
        "solver_fallback_rate": stats["fallback_rate"],
        "solver_success_rate": stats["solver_success_rate"],
    }


def evaluate_sensitivity(
    *,
    seeds: Sequence[int],
    horizons: Sequence[int],
    solver_budgets: Sequence[float],
    generator_config: FJSPGeneratorConfig | None = None,
    setup_time: float = 1.0,
    bootstrap: int = 2_000,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str | bool]],
]:
    if not seeds:
        raise ValueError("at least one seed is required")
    grid = sensitivity_grid(horizons, solver_budgets)
    generator = generator_config or FJSPGeneratorConfig(n_jobs=12, n_machines=5)
    generator.validate()
    raw: list[dict[str, float | int | str]] = []
    for horizon, budget in grid:
        for seed in seeds:
            raw.append(
                run_variant(
                    seed=int(seed),
                    generator_config=generator,
                    setup_time=setup_time,
                    horizon=horizon,
                    solver_seconds=budget,
                )
            )

    summary: list[dict[str, float | int | str | bool]] = []
    for index, (horizon, budget) in enumerate(grid):
        label = variant_name(horizon, budget)
        rows = [row for row in raw if row["policy"] == label]
        wtt = bootstrap_mean_ci(
            [float(row["weighted_tardiness"]) for row in rows],
            n_bootstrap=bootstrap,
            seed=91_000 + index,
        )
        latency = bootstrap_mean_ci(
            [float(row["mean_decision_time_ms"]) for row in rows],
            n_bootstrap=bootstrap,
            seed=92_000 + index,
        )
        summary.append(
            {
                "policy": label,
                "n_seeds": len(rows),
                "weighted_tardiness_mean": wtt.mean,
                "weighted_tardiness_std": wtt.std,
                "weighted_tardiness_ci_low": wtt.ci_low,
                "weighted_tardiness_ci_high": wtt.ci_high,
                "mean_decision_time_ms_mean": latency.mean,
                "mean_decision_time_ms_ci_low": latency.ci_low,
                "mean_decision_time_ms_ci_high": latency.ci_high,
                "solver_fallback_rate_mean": float(
                    np.mean([float(row["solver_fallback_rate"]) for row in rows])
                ),
                "solver_success_rate_mean": float(
                    np.mean([float(row["solver_success_rate"]) for row in rows])
                ),
                "cpsat_horizon": horizon,
                "solver_budget_ms": 1_000.0 * budget,
                "pareto_optimal": False,
            }
        )
    return raw, mark_pareto(summary)


def mark_pareto(
    rows: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str | bool]]:
    for row in rows:
        quality = float(row["weighted_tardiness_mean"])
        latency = float(row["mean_decision_time_ms_mean"])
        dominated = any(
            other is not row
            and float(other["weighted_tardiness_mean"]) <= quality
            and float(other["mean_decision_time_ms_mean"]) <= latency
            and (
                float(other["weighted_tardiness_mean"]) < quality
                or float(other["mean_decision_time_ms_mean"]) < latency
            )
            for other in rows
        )
        row["pareto_optimal"] = not dominated
    return rows


def compare_to_reference(
    raw_rows: list[dict[str, float | int | str]],
    *,
    reference_policy: str,
    bootstrap: int = 2_000,
    permutations: int = 5_000,
) -> list[dict[str, float | int | str]]:
    policies = sorted({str(row["policy"]) for row in raw_rows})
    if reference_policy not in policies:
        raise ValueError("reference policy is absent from sensitivity results")
    by_policy = {
        policy: {int(row["seed"]): row for row in raw_rows if row["policy"] == policy}
        for policy in policies
    }
    reference_seeds = set(by_policy[reference_policy])
    output: list[dict[str, float | int | str]] = []
    for index, candidate in enumerate(policy for policy in policies if policy != reference_policy):
        if set(by_policy[candidate]) != reference_seeds:
            raise ValueError("sensitivity variants do not share identical seed sets")
        seeds = sorted(reference_seeds)
        for metric, lower_is_better in (
            ("weighted_tardiness", True),
            ("mean_decision_time_ms", True),
        ):
            if lower_is_better:
                differences = [
                    float(by_policy[reference_policy][seed][metric])
                    - float(by_policy[candidate][seed][metric])
                    for seed in seeds
                ]
            estimate = paired_estimate(
                differences,
                n_bootstrap=bootstrap,
                n_permutations=permutations,
                seed=93_000 + index * 100 + (0 if metric == "weighted_tardiness" else 1),
            )
            output.append(
                {
                    "candidate": candidate,
                    "reference": reference_policy,
                    "metric": metric,
                    "mean_improvement": estimate.mean_difference,
                    "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high,
                    "p_value": estimate.p_value,
                    "effect_size_dz": estimate.effect_size_dz,
                    "probability_of_superiority": estimate.probability_of_superiority,
                    "n_pairs": estimate.n_pairs,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Phase-5 FJSP CP-SAT operating points.")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=41000)
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--solver-seconds", action="append", type=float, dest="budgets")
    parser.add_argument("--reference-horizon", type=int, default=12)
    parser.add_argument("--reference-solver-seconds", type=float, default=0.10)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--setup-time", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--permutations", type=int, default=5_000)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--comparisons-output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.seed_start < 41000 or args.seed_start + args.seeds > 42000:
        parser.error("Phase-5 OR validation seeds must remain in 41000-41999")
    horizons = tuple(args.horizons or DEFAULT_HORIZONS)
    budgets = tuple(args.budgets or DEFAULT_SOLVER_BUDGETS)
    grid = sensitivity_grid(horizons, budgets)
    reference = variant_name(args.reference_horizon, args.reference_solver_seconds)
    if reference not in {variant_name(h, b) for h, b in grid}:
        parser.error("reference configuration must be included in the grid")
    raw, summary = evaluate_sensitivity(
        seeds=tuple(range(args.seed_start, args.seed_start + args.seeds)),
        horizons=horizons,
        solver_budgets=budgets,
        generator_config=FJSPGeneratorConfig(
            n_jobs=args.jobs,
            n_machines=args.machines,
            n_families=4,
            operations_min=2,
            operations_max=4,
            eligible_machines_min=1,
            eligible_machines_max=min(3, args.machines),
        ),
        setup_time=args.setup_time,
        bootstrap=args.bootstrap,
    )
    comparisons = compare_to_reference(
        raw,
        reference_policy=reference,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
    )
    write_csv(args.raw_output, raw)
    write_csv(args.summary_output, summary)
    write_csv(args.comparisons_output, comparisons)
    print("FJSP CP-SAT sensitivity ranking:")
    for row in sorted(summary, key=lambda item: float(item["weighted_tardiness_mean"])):
        marker = "PARETO" if bool(row["pareto_optimal"]) else ""
        print(
            f"  {row['policy']}: WTT={float(row['weighted_tardiness_mean']):.3f} "
            f"latency={float(row['mean_decision_time_ms_mean']):.3f} ms "
            f"fallback={100.0 * float(row['solver_fallback_rate_mean']):.2f}% {marker}"
        )


if __name__ == "__main__":
    main()
