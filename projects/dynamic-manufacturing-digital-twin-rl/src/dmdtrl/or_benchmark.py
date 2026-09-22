from __future__ import annotations

import argparse
from pathlib import Path

from dmdtrl import experiments
from dmdtrl.dispatch import DispatchRule
from dmdtrl.env import EnvConfig
from dmdtrl.or_experiments import evaluate_cpsat_policy
from dmdtrl.or_policy import CPSATConfig, RollingHorizonCPSATPolicy
from dmdtrl.research import DEFAULT_COMPARISON_METRICS, fixed_policies


def main() -> None:  # pragma: no cover - CLI smoke-tested in GitHub Actions
    parser = argparse.ArgumentParser(
        description="Benchmark rolling-horizon CP-SAT against fixed dispatching rules."
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--solver-seconds", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--raw-output", type=Path, default=Path("results/or_runs.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/or_summary.csv"))
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        default=Path("results/cpsat_comparisons.csv"),
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.solver_seconds <= 0:
        parser.error("--solver-seconds must be positive")

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    env_config = EnvConfig()

    baseline_rows = experiments.evaluate_policies(
        fixed_policies(),
        seeds,
        env_config,
    )
    cpsat = RollingHorizonCPSATPolicy(
        CPSATConfig(
            max_jobs=args.horizon,
            time_limit_s=args.solver_seconds,
        )
    )
    cpsat_rows = evaluate_cpsat_policy(cpsat, seeds, env_config)

    rows = [*baseline_rows, *cpsat_rows]
    summaries = experiments.summarize_runs(
        rows,
        n_bootstrap=args.bootstrap,
    )
    comparisons = experiments.compare_candidate_to_baselines(
        rows,
        candidate=cpsat.name,
        baselines=[rule.name for rule in DispatchRule],
        metrics=DEFAULT_COMPARISON_METRICS,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
    )

    experiments.write_csv(rows, args.raw_output)
    experiments.write_csv(summaries, args.summary_output)
    experiments.write_csv(comparisons, args.comparisons_output)

    print("Weighted-tardiness ranking (lower is better):")
    for row in sorted(summaries, key=lambda item: float(item["weighted_tardiness_mean"])):
        print(
            f"  {row['policy']:<24} "
            f"mean={float(row['weighted_tardiness_mean']):.3f} "
            f"decision={float(row['mean_decision_time_ms_mean']):.3f} ms"
        )

    print(f"Seed-level OR benchmark runs saved to {args.raw_output}")
    print(f"OR benchmark summaries saved to {args.summary_output}")
    print(f"Paired CP-SAT comparisons saved to {args.comparisons_output}")


if __name__ == "__main__":  # pragma: no cover
    main()
