from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dmdtrl.fjsp_baselines import earliest_due_date_action, shortest_processing_action
from dmdtrl.fjsp_env import FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_models import FJSPInstance
from dmdtrl.fjsp_optimization import FJSPCPSATConfig, FJSPRollingHorizonCPSAT
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator
from dmdtrl.statistics import bootstrap_mean_ci, paired_estimate


@dataclass(slots=True, frozen=True)
class FJSPEvaluationConfig:
    seeds: int = 10
    seed_start: int = 40000
    default_setup_time: float = 1.0
    bootstrap: int = 2_000
    permutations: int = 5_000

    def validate(self) -> None:
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
        if self.seed_start < 40000:
            raise ValueError("Phase-5 evaluation seeds must start at 40000 or later")
        if self.default_setup_time < 0.0:
            raise ValueError("default_setup_time must be non-negative")
        if self.bootstrap <= 0 or self.permutations <= 0:
            raise ValueError("bootstrap and permutations must be positive")


def phase5_seed_regime(seed: int) -> str:
    if 40000 <= seed < 41000:
        return "development"
    if 41000 <= seed < 42000:
        return "validation"
    if seed >= 42000:
        return "final"
    raise ValueError("Phase-5 seed must be >= 40000")


def instance_fingerprint(instance: FJSPInstance) -> str:
    payload = json.dumps(asdict(instance), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_selector(
    instance: FJSPInstance,
    selector,
    *,
    setup_time: float,
    policy_name: str,
) -> dict[str, float | str]:
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=setup_time)
    decision_times: list[float] = []
    while not simulator.terminated:
        started = perf_counter()
        action = selector(simulator)
        decision_times.append(perf_counter() - started)
        simulator.step(action)
    return {
        "policy": policy_name,
        **simulator.metrics(),
        "mean_decision_time_ms": 1000.0 * float(np.mean(decision_times)),
        "fallback_rate": 0.0,
        "solver_success_rate": 1.0,
        "unique_action_fraction": float("nan"),
        "mean_feasible_actions": float("nan"),
    }


def _run_cpsat(
    instance: FJSPInstance,
    *,
    setup_time: float,
    config: FJSPCPSATConfig,
) -> dict[str, float | str]:
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=setup_time)
    controller = FJSPRollingHorizonCPSAT(config)
    while not simulator.terminated:
        decision = controller.choose(simulator)
        simulator.step(decision.action)
    stats = controller.stats()
    return {
        "policy": "ROLLING_HORIZON_CPSAT",
        **simulator.metrics(),
        "mean_decision_time_ms": stats["mean_solve_time_ms"],
        "fallback_rate": stats["fallback_rate"],
        "solver_success_rate": stats["solver_success_rate"],
        "unique_action_fraction": float("nan"),
        "mean_feasible_actions": float("nan"),
    }


def _run_maskable_model(
    model: Any,
    *,
    seed: int,
    expected_fingerprint: str,
    env_config: FJSPEnvConfig,
) -> dict[str, float | str]:
    env = FlexibleJobShopEnv(env_config)
    observation, _ = env.reset(seed=seed)
    if env.simulator is None:
        raise RuntimeError("FJSP environment did not create a simulator")
    observed_fingerprint = instance_fingerprint(env.simulator.instance)
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Maskable PPO environment generated a different FJSP instance for the common seed"
        )

    decision_times: list[float] = []
    terminated = False
    while not terminated:
        mask = env.action_masks()
        if not bool(mask.any()):
            raise RuntimeError("non-terminal Maskable PPO state has no feasible action")
        started = perf_counter()
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        decision_times.append(perf_counter() - started)
        action_id = int(np.asarray(action).item())
        if not bool(mask[action_id]):
            raise RuntimeError("Maskable PPO predicted an infeasible action")
        observation, _, terminated, truncated, _ = env.step(action_id)
        if truncated:
            raise RuntimeError("FJSP evaluation episode unexpectedly truncated")

    metrics = env.simulator.metrics()
    trace = env.decision_trace_records()
    unique_fraction = len({int(row["action_id"]) for row in trace}) / max(len(trace), 1)
    mean_feasible = float(
        np.mean([int(row["feasible_action_count"]) for row in trace])
    )
    result: dict[str, float | str] = {
        "policy": "MASKABLE_PPO",
        **metrics,
        "mean_decision_time_ms": 1000.0 * float(np.mean(decision_times)),
        "fallback_rate": 0.0,
        "solver_success_rate": 1.0,
        "unique_action_fraction": unique_fraction,
        "mean_feasible_actions": mean_feasible,
    }
    env.close()
    return result


def evaluate_panel(
    evaluation_config: FJSPEvaluationConfig | None = None,
    *,
    generator_config: FJSPGeneratorConfig | None = None,
    cpsat_config: FJSPCPSATConfig | None = None,
    maskable_model: Any | None = None,
) -> list[dict[str, float | int | str]]:
    evaluation = evaluation_config or FJSPEvaluationConfig()
    evaluation.validate()
    generator = generator_config or FJSPGeneratorConfig()
    generator.validate()
    cpsat = cpsat_config or FJSPCPSATConfig()
    cpsat.validate()
    env_config = FJSPEnvConfig(
        generator=generator,
        default_setup_time=evaluation.default_setup_time,
    )

    rows: list[dict[str, float | int | str]] = []
    for seed in range(evaluation.seed_start, evaluation.seed_start + evaluation.seeds):
        regime = phase5_seed_regime(seed)
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator)
        fingerprint = instance_fingerprint(instance)
        results = [
            _run_selector(
                instance,
                shortest_processing_action,
                setup_time=evaluation.default_setup_time,
                policy_name="SHORTEST_PROCESSING",
            ),
            _run_selector(
                instance,
                earliest_due_date_action,
                setup_time=evaluation.default_setup_time,
                policy_name="EARLIEST_DUE_DATE",
            ),
            _run_cpsat(
                instance,
                setup_time=evaluation.default_setup_time,
                config=cpsat,
            ),
        ]
        if maskable_model is not None:
            results.append(
                _run_maskable_model(
                    maskable_model,
                    seed=seed,
                    expected_fingerprint=fingerprint,
                    env_config=env_config,
                )
            )
        for result in results:
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": regime,
                    "instance_sha256": fingerprint,
                    **result,
                }
            )
    return rows


def summarize_panel(
    rows: list[dict[str, float | int | str]],
    *,
    bootstrap: int = 2_000,
) -> list[dict[str, float | int | str]]:
    policies = sorted({str(row["policy"]) for row in rows})
    summary: list[dict[str, float | int | str]] = []
    for index, policy in enumerate(policies):
        selected = [row for row in rows if row["policy"] == policy]
        wtt = [float(row["weighted_tardiness"]) for row in selected]
        estimate = bootstrap_mean_ci(wtt, n_bootstrap=bootstrap, seed=70_000 + index)
        summary.append(
            {
                "policy": policy,
                "n_seeds": len(selected),
                "weighted_tardiness_mean": estimate.mean,
                "weighted_tardiness_std": estimate.std,
                "weighted_tardiness_ci_low": estimate.ci_low,
                "weighted_tardiness_ci_high": estimate.ci_high,
                "makespan_mean": float(np.mean([float(row["makespan"]) for row in selected])),
                "mean_flow_time": float(
                    np.mean([float(row["mean_flow_time"]) for row in selected])
                ),
                "mean_decision_time_ms": float(
                    np.mean([float(row["mean_decision_time_ms"]) for row in selected])
                ),
                "fallback_rate": float(
                    np.mean([float(row["fallback_rate"]) for row in selected])
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (float(row["weighted_tardiness_mean"]), str(row["policy"])),
    )


def compare_candidate(
    rows: list[dict[str, float | int | str]],
    *,
    candidate: str,
    metric: str = "weighted_tardiness",
    bootstrap: int = 2_000,
    permutations: int = 5_000,
) -> list[dict[str, float | int | str]]:
    policies = sorted({str(row["policy"]) for row in rows})
    if candidate not in policies:
        raise ValueError(f"candidate policy {candidate!r} is absent")
    by_policy = {
        policy: {int(row["seed"]): row for row in rows if row["policy"] == policy}
        for policy in policies
    }
    candidate_seeds = set(by_policy[candidate])
    comparisons: list[dict[str, float | int | str]] = []
    for index, baseline in enumerate(policy for policy in policies if policy != candidate):
        baseline_seeds = set(by_policy[baseline])
        if baseline_seeds != candidate_seeds:
            raise ValueError("paired policy seed sets are not identical")
        seeds = sorted(candidate_seeds)
        differences = np.asarray(
            [
                float(by_policy[baseline][seed][metric])
                - float(by_policy[candidate][seed][metric])
                for seed in seeds
            ],
            dtype=float,
        )
        estimate = paired_estimate(
            differences,
            n_bootstrap=bootstrap,
            n_permutations=permutations,
            seed=80_000 + index,
        )
        baseline_mean = float(
            np.mean([float(by_policy[baseline][seed][metric]) for seed in seeds])
        )
        comparisons.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "mean_improvement": estimate.mean_difference,
                "percent_improvement": (
                    100.0 * estimate.mean_difference / baseline_mean
                    if abs(baseline_mean) > 1e-12
                    else 0.0
                ),
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "p_value": estimate.p_value,
                "effect_size_dz": estimate.effect_size_dz,
                "probability_of_superiority": estimate.probability_of_superiority,
                "n_pairs": estimate.n_pairs,
            }
        )
    return comparisons


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_maskable_ppo(path: str | Path, *, device: str = "cpu") -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError("sb3-contrib is required to evaluate a Maskable PPO model") from exc
    return MaskablePPO.load(str(path), device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase-5 FJSP controllers on common seeds.")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--operations-min", type=int, default=2)
    parser.add_argument("--operations-max", type=int, default=4)
    parser.add_argument("--eligible-max", type=int, default=3)
    parser.add_argument("--setup-time", type=float, default=1.0)
    parser.add_argument("--cpsat-horizon", type=int, default=8)
    parser.add_argument("--cpsat-seconds", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--permutations", type=int, default=5_000)
    parser.add_argument("--raw-output", default="results/fjsp_eval_runs.csv")
    parser.add_argument("--summary-output", default="results/fjsp_eval_summary.csv")
    parser.add_argument("--comparisons-output", default="results/fjsp_eval_comparisons.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation = FJSPEvaluationConfig(
        seeds=args.seeds,
        seed_start=args.seed_start,
        default_setup_time=args.setup_time,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
    )
    generator = FJSPGeneratorConfig(
        n_jobs=args.jobs,
        n_machines=args.machines,
        n_families=4,
        operations_min=args.operations_min,
        operations_max=args.operations_max,
        eligible_machines_min=1,
        eligible_machines_max=min(args.eligible_max, args.machines),
    )
    cpsat = FJSPCPSATConfig(
        job_horizon=args.cpsat_horizon,
        solver_seconds=args.cpsat_seconds,
    )
    model = load_maskable_ppo(args.model, device=args.device) if args.model else None
    rows = evaluate_panel(
        evaluation,
        generator_config=generator,
        cpsat_config=cpsat,
        maskable_model=model,
    )
    summary = summarize_panel(rows, bootstrap=evaluation.bootstrap)
    candidate = "MASKABLE_PPO" if model is not None else "ROLLING_HORIZON_CPSAT"
    comparisons = compare_candidate(
        rows,
        candidate=candidate,
        bootstrap=evaluation.bootstrap,
        permutations=evaluation.permutations,
    )
    write_csv(args.raw_output, rows)
    write_csv(args.summary_output, summary)
    write_csv(args.comparisons_output, comparisons)
    for rank, row in enumerate(summary, start=1):
        print(
            f"{rank}. {row['policy']}: WTT={float(row['weighted_tardiness_mean']):.3f} "
            f"latency={float(row['mean_decision_time_ms']):.3f} ms"
        )


if __name__ == "__main__":
    main()
