from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dmdtrl.fjsp_env import FJSPEnvConfig
from dmdtrl.fjsp_evaluate import (
    compare_candidate,
    instance_fingerprint,
    summarize_panel,
    write_csv,
)
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_hyperheuristic_env import FlexibleJobShopHyperHeuristicEnv
from dmdtrl.fjsp_models import FJSPInstance
from dmdtrl.fjsp_operators import FJSPOperator, select_operator_action
from dmdtrl.fjsp_optimization import FJSPCPSATConfig, FJSPRollingHorizonCPSAT
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


@dataclass(slots=True, frozen=True)
class FJSPHyperHeuristicDevelopmentConfig:
    seeds: int = 10
    seed_start: int = 40000
    default_setup_time: float = 1.0
    bootstrap: int = 2_000
    permutations: int = 5_000

    def validate(self) -> None:
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
        if self.seed_start < 40000:
            raise ValueError("hyper-heuristic development seeds must start at 40000")
        if self.seed_start + self.seeds > 41000:
            raise ValueError("development evaluator must not access seed 41000 or later")
        if self.default_setup_time < 0.0:
            raise ValueError("default_setup_time must be non-negative")
        if self.bootstrap <= 0 or self.permutations <= 0:
            raise ValueError("bootstrap and permutations must be positive")


def _run_operator(
    instance: FJSPInstance,
    operator: FJSPOperator,
    *,
    setup_time: float,
) -> dict[str, float | str]:
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=setup_time)
    decision_times: list[float] = []
    while not simulator.terminated:
        started = perf_counter()
        action = select_operator_action(simulator, operator)
        decision_times.append(perf_counter() - started)
        simulator.step(action)
    return {
        "policy": operator.name,
        **simulator.metrics(),
        "mean_decision_time_ms": 1000.0 * float(np.mean(decision_times)),
        "fallback_rate": 0.0,
        "solver_success_rate": 1.0,
        "unique_operator_fraction": 1.0 / len(FJSPOperator),
        "mean_underlying_feasible_actions": float("nan"),
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
        "unique_operator_fraction": float("nan"),
        "mean_underlying_feasible_actions": float("nan"),
    }


def _run_ppo(
    model: Any,
    *,
    seed: int,
    expected_fingerprint: str,
    env_config: FJSPEnvConfig,
) -> dict[str, float | str]:
    env = FlexibleJobShopHyperHeuristicEnv(env_config)
    observation, _ = env.reset(seed=seed)
    if env.simulator is None:
        raise RuntimeError("hyper-heuristic environment did not create a simulator")
    observed_fingerprint = instance_fingerprint(env.simulator.instance)
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError("PPO hyper-heuristic generated a different common-seed instance")

    decision_times: list[float] = []
    terminated = False
    while not terminated:
        started = perf_counter()
        action, _ = model.predict(observation, deterministic=True)
        decision_times.append(perf_counter() - started)
        operator_id = int(np.asarray(action).item())
        if not env.action_space.contains(operator_id):
            raise RuntimeError("PPO hyper-heuristic predicted an invalid operator ID")
        observation, _, terminated, truncated, _ = env.step(operator_id)
        if truncated:
            raise RuntimeError("hyper-heuristic evaluation episode unexpectedly truncated")

    if env.simulator is None:
        raise RuntimeError("hyper-heuristic simulator disappeared during evaluation")
    metrics = env.simulator.metrics()
    trace = env.decision_trace_records()
    selected_operators = {int(row["operator_action_id"]) for row in trace}
    unique_operator_fraction = len(selected_operators) / len(FJSPOperator)
    mean_feasible = float(
        np.mean([int(row["underlying_feasible_action_count"]) for row in trace])
    )
    result: dict[str, float | str] = {
        "policy": "PPO_HYPER_HEURISTIC",
        **metrics,
        "mean_decision_time_ms": 1000.0 * float(np.mean(decision_times)),
        "fallback_rate": 0.0,
        "solver_success_rate": 1.0,
        "unique_operator_fraction": unique_operator_fraction,
        "mean_underlying_feasible_actions": mean_feasible,
    }
    env.close()
    return result


def evaluate_development_panel(
    evaluation_config: FJSPHyperHeuristicDevelopmentConfig | None = None,
    *,
    generator_config: FJSPGeneratorConfig | None = None,
    cpsat_config: FJSPCPSATConfig | None = None,
    ppo_model: Any | None = None,
) -> list[dict[str, float | int | str]]:
    evaluation = evaluation_config or FJSPHyperHeuristicDevelopmentConfig()
    evaluation.validate()
    generator = generator_config or FJSPGeneratorConfig()
    generator.validate()
    cpsat = cpsat_config or FJSPCPSATConfig(job_horizon=4, solver_seconds=0.10)
    cpsat.validate()
    env_config = FJSPEnvConfig(
        generator=generator,
        default_setup_time=evaluation.default_setup_time,
    )

    rows: list[dict[str, float | int | str]] = []
    for seed in range(evaluation.seed_start, evaluation.seed_start + evaluation.seeds):
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator)
        fingerprint = instance_fingerprint(instance)
        results = [
            _run_operator(
                instance,
                operator,
                setup_time=evaluation.default_setup_time,
            )
            for operator in FJSPOperator
        ]
        results.append(
            _run_cpsat(
                instance,
                setup_time=evaluation.default_setup_time,
                config=cpsat,
            )
        )
        if ppo_model is not None:
            results.append(
                _run_ppo(
                    ppo_model,
                    seed=seed,
                    expected_fingerprint=fingerprint,
                    env_config=env_config,
                )
            )
        for result in results:
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "development",
                    "instance_sha256": fingerprint,
                    **result,
                }
            )
    return rows


def load_ppo(path: str | Path, *, device: str = "cpu") -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to evaluate PPO") from exc
    return PPO.load(str(path), device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the FJSP PPO hyper-heuristic on development seeds only."
    )
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
    parser.add_argument("--cpsat-horizon", type=int, default=4)
    parser.add_argument("--cpsat-seconds", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--permutations", type=int, default=5_000)
    parser.add_argument("--raw-output", default="results/fjsp_hh_dev_runs.csv")
    parser.add_argument("--summary-output", default="results/fjsp_hh_dev_summary.csv")
    parser.add_argument("--comparisons-output", default="results/fjsp_hh_dev_comparisons.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation = FJSPHyperHeuristicDevelopmentConfig(
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
    model = load_ppo(args.model, device=args.device) if args.model else None
    rows = evaluate_development_panel(
        evaluation,
        generator_config=generator,
        cpsat_config=cpsat,
        ppo_model=model,
    )
    summary = summarize_panel(rows, bootstrap=evaluation.bootstrap)
    candidate = "PPO_HYPER_HEURISTIC" if model is not None else "ROLLING_HORIZON_CPSAT"
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
