from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from dmdtrl.dispatch import DispatchRule
from dmdtrl.env import DynamicManufacturingEnv, EnvConfig


def run_fixed_rule(rule: DispatchRule, seed: int, config: EnvConfig) -> dict[str, float]:
    env = DynamicManufacturingEnv(config=config)
    env.reset(seed=seed)
    terminated = False
    while not terminated:
        _, _, terminated, truncated, _ = env.step(int(rule))
        if truncated:
            break
    return env.metrics()


def benchmark(
    seeds: list[int],
    config: EnvConfig | None = None,
) -> list[dict[str, float | str]]:
    cfg = config or EnvConfig()
    rows: list[dict[str, float | str]] = []
    for rule in DispatchRule:
        runs = [run_fixed_rule(rule, seed, cfg) for seed in seeds]
        rows.append(
            {
                "policy": rule.name,
                "mean_waiting_time": mean(r["mean_waiting_time"] for r in runs),
                "weighted_tardiness": mean(r["weighted_tardiness"] for r in runs),
                "total_setup_time": mean(r["total_setup_time"] for r in runs),
                "on_time_rate": mean(r["on_time_rate"] for r in runs),
                "utilization": mean(r["utilization"] for r in runs),
                "makespan": mean(r["makespan"] for r in runs),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deterministic dispatching rules.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of common random seeds.")
    parser.add_argument("--output", type=Path, default=Path("results/baselines.csv"))
    args = parser.parse_args()

    rows = benchmark(list(range(args.seeds)))
    write_csv(rows, args.output)
    for row in rows:
        print(row)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
