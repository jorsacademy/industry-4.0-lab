from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from dmdtrl.policies import DispatchPolicy
from dmdtrl.statistics import bootstrap_mean_ci, paired_estimate

if TYPE_CHECKING:
    from dmdtrl.env import EnvConfig


METRIC_DIRECTIONS: dict[str, str] = {
    "mean_waiting_time": "min",
    "total_tardiness": "min",
    "weighted_tardiness": "min",
    "total_setup_time": "min",
    "total_repair_time": "min",
    "makespan": "min",
    "on_time_rate": "max",
    "utilization": "max",
    "mean_quality_risk": "min",
    "mean_decision_time_ms": "min",
}

EnvFactory = Callable[[Any], Any]


def run_policy(
    policy: DispatchPolicy,
    seed: int,
    config: EnvConfig | None = None,
    *,
    env_factory: EnvFactory | None = None,
) -> dict[str, float | int | str]:
    """Run one policy on one stochastic seed and return seed-level KPIs."""
    if env_factory is None:
        from dmdtrl.env import DynamicManufacturingEnv, EnvConfig

        cfg = config or EnvConfig()
        env = DynamicManufacturingEnv(config=cfg)
    else:
        env = env_factory(config)

    observation, _ = env.reset(seed=seed)
    terminated = False
    decision_times: list[float] = []
    while not terminated:
        started = perf_counter()
        action = policy.act(observation)
        decision_times.append(perf_counter() - started)
        observation, _, terminated, truncated, _ = env.step(action)
        if truncated:
            break

    return {
        "policy": policy.name,
        "seed": seed,
        **env.metrics(),
        "mean_decision_time_ms": 1_000.0 * mean(decision_times) if decision_times else 0.0,
    }


def evaluate_policies(
    policies: Iterable[DispatchPolicy],
    seeds: Iterable[int],
    config: EnvConfig | None = None,
    *,
    env_factory: EnvFactory | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one evaluation seed is required")
    for policy in policies:
        for seed in seed_list:
            rows.append(run_policy(policy, seed, config, env_factory=env_factory))
    return rows


def summarize_runs(
    rows: list[dict[str, float | int | str]],
    *,
    metrics: Iterable[str] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    seed: int = 12_345,
) -> list[dict[str, float | int | str]]:
    if not rows:
        raise ValueError("rows must not be empty")
    selected_metrics = list(metrics or METRIC_DIRECTIONS)
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)

    summaries: list[dict[str, float | int | str]] = []
    for policy_index, (policy, policy_rows) in enumerate(sorted(grouped.items())):
        summary: dict[str, float | int | str] = {"policy": policy, "n_seeds": len(policy_rows)}
        for metric_index, metric in enumerate(selected_metrics):
            values = [float(row[metric]) for row in policy_rows]
            estimate = bootstrap_mean_ci(
                values,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                seed=seed + policy_index * 1_000 + metric_index,
            )
            summary[f"{metric}_mean"] = estimate.mean
            summary[f"{metric}_std"] = estimate.std
            summary[f"{metric}_ci_low"] = estimate.ci_low
            summary[f"{metric}_ci_high"] = estimate.ci_high
        summaries.append(summary)
    return summaries


def paired_comparison(
    rows: list[dict[str, float | int | str]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
    direction: str | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    n_permutations: int = 10_000,
    seed: int = 12_345,
) -> dict[str, float | int | str]:
    comparison_direction = direction or METRIC_DIRECTIONS.get(metric)
    if comparison_direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")

    by_policy: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        policy = str(row["policy"])
        if policy in {candidate, baseline}:
            by_policy[policy][int(row["seed"])] = float(row[metric])

    candidate_values = by_policy.get(candidate, {})
    baseline_values = by_policy.get(baseline, {})
    common = sorted(set(candidate_values) & set(baseline_values))
    if not common:
        raise ValueError(f"no paired seeds for {candidate!r} and {baseline!r}")

    candidate_array = np.array([candidate_values[s] for s in common], dtype=float)
    baseline_array = np.array([baseline_values[s] for s in common], dtype=float)
    if comparison_direction == "min":
        improvement = baseline_array - candidate_array
    else:
        improvement = candidate_array - baseline_array

    estimate = paired_estimate(
        improvement,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
    )
    baseline_mean = float(np.mean(baseline_array))
    percent = (
        100.0 * estimate.mean_difference / abs(baseline_mean)
        if baseline_mean != 0.0
        else np.nan
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "direction": comparison_direction,
        "n_pairs": estimate.n_pairs,
        "candidate_mean": float(np.mean(candidate_array)),
        "baseline_mean": baseline_mean,
        "mean_improvement": estimate.mean_difference,
        "improvement_ci_low": estimate.ci_low,
        "improvement_ci_high": estimate.ci_high,
        "percent_improvement": float(percent),
        "p_value": estimate.p_value,
        "effect_size_dz": estimate.effect_size_dz,
        "probability_of_superiority": estimate.probability_of_superiority,
    }


def compare_candidate_to_baselines(
    rows: list[dict[str, float | int | str]],
    *,
    candidate: str,
    baselines: Iterable[str],
    metrics: Iterable[str] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    n_permutations: int = 10_000,
    seed: int = 12_345,
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    selected_metrics = list(metrics or METRIC_DIRECTIONS)
    for baseline_index, baseline in enumerate(baselines):
        if baseline == candidate:
            continue
        for metric_index, metric in enumerate(selected_metrics):
            results.append(
                paired_comparison(
                    rows,
                    candidate=candidate,
                    baseline=baseline,
                    metric=metric,
                    confidence=confidence,
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    seed=seed + baseline_index * 1_000 + metric_index,
                )
            )
    return results


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
