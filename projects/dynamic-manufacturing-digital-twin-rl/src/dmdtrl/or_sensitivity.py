from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from dmdtrl.env import EnvConfig
from dmdtrl.experiments import (
    METRIC_DIRECTIONS,
    compare_candidate_to_baselines,
    summarize_runs,
    write_csv,
)
from dmdtrl.or_experiments import evaluate_cpsat_policy
from dmdtrl.or_policy import CPSATConfig, RollingHorizonCPSATPolicy

DEFAULT_HORIZONS = (4, 8, 12)
DEFAULT_SOLVER_BUDGETS = (0.02, 0.05, 0.10)
SENSITIVITY_METRICS = ("weighted_tardiness", "mean_decision_time_ms")
SENSITIVITY_SUMMARY_METRICS = (
    *METRIC_DIRECTIONS,
    "solver_fallback_rate",
    "solver_success_rate",
)


def variant_name(horizon: int, solver_seconds: float) -> str:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if solver_seconds <= 0:
        raise ValueError("solver_seconds must be positive")
    budget_us = int(round(solver_seconds * 1_000_000))
    return f"CP_SAT_H{horizon}_B{budget_us}US"


def sensitivity_grid(
    horizons: Sequence[int],
    solver_budgets: Sequence[float],
) -> list[tuple[int, float]]:
    if not horizons or not solver_budgets:
        raise ValueError("horizon and solver-budget grids must be non-empty")
    if any(value <= 0 for value in horizons):
        raise ValueError("all horizons must be positive")
    if any(value <= 0 for value in solver_budgets):
        raise ValueError("all solver budgets must be positive")

    grid = [(int(horizon), float(budget)) for horizon in horizons for budget in solver_budgets]
    names = [variant_name(horizon, budget) for horizon, budget in grid]
    if len(set(names)) != len(names):
        raise ValueError("sensitivity grid contains duplicate configurations")
    if len(grid) < 2:
        raise ValueError("sensitivity grid must contain at least two configurations")
    return grid


def _mark_pareto(
    summary_rows: list[dict[str, float | int | str | bool]],
) -> list[dict[str, float | int | str | bool]]:
    for row in summary_rows:
        quality = float(row["weighted_tardiness_mean"])
        latency = float(row["mean_decision_time_ms_mean"])
        dominated = False
        for other in summary_rows:
            if other is row:
                continue
            other_quality = float(other["weighted_tardiness_mean"])
            other_latency = float(other["mean_decision_time_ms_mean"])
            no_worse = other_quality <= quality and other_latency <= latency
            strictly_better = other_quality < quality or other_latency < latency
            if no_worse and strictly_better:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return summary_rows


def evaluate_sensitivity(
    seeds: Iterable[int],
    horizons: Sequence[int],
    solver_budgets: Sequence[float],
    config: EnvConfig | None = None,
    *,
    n_bootstrap: int = 5_000,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str | bool]],
]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one evaluation seed is required")
    grid = sensitivity_grid(horizons, solver_budgets)
    env_config = config or EnvConfig()

    raw_rows: list[dict[str, float | int | str]] = []
    config_by_policy: dict[str, tuple[int, float]] = {}
    for horizon, budget in grid:
        label = variant_name(horizon, budget)
        policy = RollingHorizonCPSATPolicy(
            CPSATConfig(max_jobs=horizon, time_limit_s=budget)
        )
        rows = evaluate_cpsat_policy(policy, seed_list, env_config)
        config_by_policy[label] = (horizon, budget)
        for row in rows:
            raw_rows.append(
                {
                    **row,
                    "policy": label,
                    "cpsat_horizon": horizon,
                    "solver_budget_ms": 1_000.0 * budget,
                }
            )

    summaries = summarize_runs(
        raw_rows,
        metrics=SENSITIVITY_SUMMARY_METRICS,
        n_bootstrap=n_bootstrap,
        seed=55_000,
    )
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for row in summaries:
        label = str(row["policy"])
        horizon, budget = config_by_policy[label]
        summary_rows.append(
            {
                **row,
                "cpsat_horizon": horizon,
                "solver_budget_ms": 1_000.0 * budget,
                "pareto_optimal": False,
            }
        )
    return raw_rows, _mark_pareto(summary_rows)


def compare_to_reference(
    raw_rows: list[dict[str, float | int | str]],
    *,
    reference_policy: str,
    n_bootstrap: int = 5_000,
    n_permutations: int = 10_000,
) -> list[dict[str, float | int | str]]:
    policies = sorted({str(row["policy"]) for row in raw_rows})
    if reference_policy not in policies:
        raise ValueError(f"reference policy {reference_policy!r} is not present in results")

    comparisons: list[dict[str, float | int | str]] = []
    for index, candidate in enumerate(policies):
        if candidate == reference_policy:
            continue
        rows = compare_candidate_to_baselines(
            raw_rows,
            candidate=candidate,
            baselines=[reference_policy],
            metrics=SENSITIVITY_METRICS,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=65_000 + index * 1_000,
        )
        comparisons.extend(rows)
    return comparisons


def main() -> None:  # pragma: no cover - CLI smoke-tested in GitHub Actions
    parser = argparse.ArgumentParser(
        description="Sweep CP-SAT horizon and online solve-budget configurations."
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument(
        "--solver-seconds",
        action="append",
        type=float,
        dest="solver_budgets",
    )
    parser.add_argument("--reference-horizon", type=int, default=12)
    parser.add_argument("--reference-solver-seconds", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("results/cpsat_sensitivity_runs.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/cpsat_sensitivity_summary.csv"),
    )
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        default=Path("results/cpsat_sensitivity_comparisons.csv"),
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    horizons = tuple(args.horizons or DEFAULT_HORIZONS)
    budgets = tuple(args.solver_budgets or DEFAULT_SOLVER_BUDGETS)
    try:
        grid = sensitivity_grid(horizons, budgets)
        reference_policy = variant_name(
            args.reference_horizon,
            args.reference_solver_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    grid_names = {variant_name(horizon, budget) for horizon, budget in grid}
    if reference_policy not in grid_names:
        parser.error("reference configuration must be included in the sensitivity grid")

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    raw_rows, summary_rows = evaluate_sensitivity(
        seeds,
        horizons,
        budgets,
        n_bootstrap=args.bootstrap,
    )
    comparisons = compare_to_reference(
        raw_rows,
        reference_policy=reference_policy,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
    )

    write_csv(raw_rows, args.raw_output)
    write_csv(summary_rows, args.summary_output)
    write_csv(comparisons, args.comparisons_output)

    print("CP-SAT sensitivity ranking (lower WTT is better):")
    for row in sorted(summary_rows, key=lambda item: float(item["weighted_tardiness_mean"])):
        marker = "PARETO" if bool(row["pareto_optimal"]) else ""
        print(
            f"  {str(row['policy']):<28} "
            f"WTT={float(row['weighted_tardiness_mean']):.3f} "
            f"decision={float(row['mean_decision_time_ms_mean']):.3f} ms "
            f"fallback={100.0 * float(row['solver_fallback_rate_mean']):.2f}% {marker}"
        )

    print(f"Sensitivity runs saved to {args.raw_output}")
    print(f"Sensitivity summaries saved to {args.summary_output}")
    print(f"Reference comparisons saved to {args.comparisons_output}")


if __name__ == "__main__":  # pragma: no cover
    main()
