from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from dmdtrl.fjsp_baselines import earliest_due_date_action, shortest_processing_action
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_optimization import FJSPCPSATConfig, FJSPRollingHorizonCPSAT
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


@dataclass(slots=True, frozen=True)
class FJSPBenchmarkConfig:
    seeds: int = 5
    seed_start: int = 40000
    default_setup_time: float = 1.5

    def validate(self) -> None:
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
        if self.seed_start < 40000:
            raise ValueError("Phase-5 development seeds must start at 40000 or later")
        if self.default_setup_time < 0.0:
            raise ValueError("default_setup_time must be non-negative")


def _run_selector(instance, selector, setup_time: float, policy_name: str) -> dict[str, float | str]:
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=setup_time)
    decision_times: list[float] = []
    while not simulator.terminated:
        started = perf_counter()
        action = selector(simulator)
        decision_times.append(perf_counter() - started)
        simulator.step(action)
    metrics = simulator.metrics()
    return {
        "policy": policy_name,
        **metrics,
        "mean_decision_time_ms": 1000.0 * sum(decision_times) / max(len(decision_times), 1),
        "fallback_rate": 0.0,
        "solver_success_rate": 1.0,
    }


def _run_cpsat(instance, setup_time: float, config: FJSPCPSATConfig) -> dict[str, float | str]:
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=setup_time)
    controller = FJSPRollingHorizonCPSAT(config)
    while not simulator.terminated:
        decision = controller.choose(simulator)
        simulator.step(decision.action)
    stats = controller.stats()
    return {
        "policy": "ROLLING_HORIZON_CPSAT",
        **simulator.metrics(),
        **stats,
        "mean_decision_time_ms": stats["mean_solve_time_ms"],
    }


def run_benchmark(
    benchmark_config: FJSPBenchmarkConfig | None = None,
    *,
    generator_config: FJSPGeneratorConfig | None = None,
    cpsat_config: FJSPCPSATConfig | None = None,
) -> list[dict[str, float | int | str]]:
    bench = benchmark_config or FJSPBenchmarkConfig()
    bench.validate()
    generator = generator_config or FJSPGeneratorConfig()
    generator.validate()
    cpsat = cpsat_config or FJSPCPSATConfig()
    cpsat.validate()

    rows: list[dict[str, float | int | str]] = []
    for seed in range(bench.seed_start, bench.seed_start + bench.seeds):
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator)
        for result in (
            _run_selector(
                instance,
                shortest_processing_action,
                bench.default_setup_time,
                "SHORTEST_PROCESSING",
            ),
            _run_selector(
                instance,
                earliest_due_date_action,
                bench.default_setup_time,
                "EARLIEST_DUE_DATE",
            ),
            _run_cpsat(instance, bench.default_setup_time, cpsat),
        ):
            rows.append({"seed": seed, **result})
    return rows


def summarize_benchmark(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | str]]:
    policies = sorted({str(row["policy"]) for row in rows})
    summary: list[dict[str, float | str]] = []
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        summary.append(
            {
                "policy": policy,
                "weighted_tardiness": float(
                    np.mean([float(row["weighted_tardiness"]) for row in selected])
                ),
                "makespan": float(np.mean([float(row["makespan"]) for row in selected])),
                "mean_flow_time": float(
                    np.mean([float(row["mean_flow_time"]) for row in selected])
                ),
                "total_setup_time": float(
                    np.mean([float(row["total_setup_time"]) for row in selected])
                ),
                "mean_decision_time_ms": float(
                    np.mean([float(row["mean_decision_time_ms"]) for row in selected])
                ),
                "fallback_rate": float(
                    np.mean([float(row["fallback_rate"]) for row in selected])
                ),
            }
        )
    return sorted(summary, key=lambda row: (float(row["weighted_tardiness"]), str(row["policy"])))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Phase-5 FJSP baselines.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--job-horizon", type=int, default=8)
    parser.add_argument("--solver-seconds", type=float, default=0.10)
    parser.add_argument("--setup-time", type=float, default=1.5)
    parser.add_argument("--raw-output", default="results/fjsp_or_runs.csv")
    parser.add_argument("--summary-output", default="results/fjsp_or_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_benchmark(
        FJSPBenchmarkConfig(
            seeds=args.seeds,
            seed_start=args.seed_start,
            default_setup_time=args.setup_time,
        ),
        generator_config=FJSPGeneratorConfig(
            n_jobs=args.jobs,
            n_machines=args.machines,
            n_families=4,
            operations_min=2,
            operations_max=4,
            eligible_machines_min=1,
            eligible_machines_max=min(3, args.machines),
        ),
        cpsat_config=FJSPCPSATConfig(
            job_horizon=args.job_horizon,
            solver_seconds=args.solver_seconds,
        ),
    )
    summary = summarize_benchmark(rows)
    write_csv(args.raw_output, rows)
    write_csv(args.summary_output, summary)
    for rank, row in enumerate(summary, start=1):
        print(
            f"{rank}. {row['policy']}: WTT={float(row['weighted_tardiness']):.3f} "
            f"latency={float(row['mean_decision_time_ms']):.3f} ms "
            f"fallback={100.0 * float(row['fallback_rate']):.2f}%"
        )


if __name__ == "__main__":
    main()
