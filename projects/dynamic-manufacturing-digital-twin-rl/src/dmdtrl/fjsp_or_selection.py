from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

VALIDATION_MIN = 41000
VALIDATION_MAX_EXCLUSIVE = 42000
REQUIRED_FIELDS = {
    "policy",
    "weighted_tardiness_mean",
    "mean_decision_time_ms_mean",
    "solver_fallback_rate_mean",
    "cpsat_horizon",
    "solver_budget_ms",
    "pareto_optimal",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as boolean")


def validate_validation_grid(
    raw_rows: list[dict[str, Any]],
    *,
    seed_start: int,
    seed_count: int,
) -> list[int]:
    if seed_count <= 0:
        raise ValueError("validation seed count must be positive")
    if seed_start < VALIDATION_MIN or seed_start + seed_count > VALIDATION_MAX_EXCLUSIVE:
        raise ValueError("FJSP OR validation seeds must remain in 41000-41999")
    if not raw_rows:
        raise ValueError("raw sensitivity results are empty")

    expected = list(range(seed_start, seed_start + seed_count))
    expected_set = set(expected)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_policy[str(row["policy"])].append(row)
    if len(by_policy) < 2:
        raise ValueError("raw sensitivity results require multiple configurations")

    fingerprints_by_seed: dict[int, set[str]] = defaultdict(set)
    for policy, rows in by_policy.items():
        observed = {int(row["seed"]) for row in rows}
        if observed != expected_set:
            raise ValueError(
                f"validation seeds for {policy!r} do not match declared range"
            )
        if len(rows) != seed_count:
            raise ValueError(f"validation rows for {policy!r} contain duplicate seeds")
        for row in rows:
            seed = int(row["seed"])
            if str(row.get("seed_regime")) != "validation":
                raise ValueError("raw sensitivity row is not labeled as validation")
            fingerprints_by_seed[seed].add(str(row["instance_sha256"]))

    for seed, fingerprints in fingerprints_by_seed.items():
        if len(fingerprints) != 1:
            raise ValueError(f"seed {seed} has inconsistent instance fingerprints across grid")
    return expected


def select_operating_point(
    summary_rows: list[dict[str, Any]],
    *,
    quality_tolerance_pct: float = 2.0,
    max_fallback_rate_pct: float = 1.0,
) -> dict[str, Any]:
    if quality_tolerance_pct < 0.0:
        raise ValueError("quality_tolerance_pct must be non-negative")
    if not 0.0 <= max_fallback_rate_pct <= 100.0:
        raise ValueError("max_fallback_rate_pct must be in [0, 100]")
    if not summary_rows:
        raise ValueError("sensitivity summary is empty")

    parsed: list[dict[str, Any]] = []
    for row in summary_rows:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"summary is missing fields: {sorted(missing)}")
        parsed_row = {
            **row,
            "weighted_tardiness_mean": float(row["weighted_tardiness_mean"]),
            "mean_decision_time_ms_mean": float(row["mean_decision_time_ms_mean"]),
            "solver_fallback_rate_mean": float(row["solver_fallback_rate_mean"]),
            "cpsat_horizon": int(float(row["cpsat_horizon"])),
            "solver_budget_ms": float(row["solver_budget_ms"]),
            "pareto_optimal": as_bool(row["pareto_optimal"]),
        }
        numeric = (
            parsed_row["weighted_tardiness_mean"],
            parsed_row["mean_decision_time_ms_mean"],
            parsed_row["solver_fallback_rate_mean"],
            parsed_row["solver_budget_ms"],
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("summary contains non-finite values")
        if (
            parsed_row["weighted_tardiness_mean"] < 0.0
            or parsed_row["mean_decision_time_ms_mean"] < 0.0
            or not 0.0 <= parsed_row["solver_fallback_rate_mean"] <= 1.0
            or parsed_row["cpsat_horizon"] <= 0
            or parsed_row["solver_budget_ms"] <= 0.0
        ):
            raise ValueError("summary contains invalid operating values")
        parsed.append(parsed_row)

    pareto = [row for row in parsed if row["pareto_optimal"]]
    if not pareto:
        raise ValueError("summary contains no Pareto-optimal configuration")
    max_fallback = max_fallback_rate_pct / 100.0
    reliable = [
        row
        for row in pareto
        if float(row["solver_fallback_rate_mean"]) <= max_fallback + 1e-12
    ]
    if not reliable:
        raise ValueError("no reliable Pareto configuration satisfies fallback limit")

    best_wtt = min(float(row["weighted_tardiness_mean"]) for row in reliable)
    threshold = best_wtt * (1.0 + quality_tolerance_pct / 100.0)
    acceptable = [
        row for row in reliable if float(row["weighted_tardiness_mean"]) <= threshold + 1e-12
    ]
    selected = min(
        acceptable,
        key=lambda row: (
            float(row["mean_decision_time_ms_mean"]),
            float(row["weighted_tardiness_mean"]),
            float(row["solver_fallback_rate_mean"]),
            float(row["solver_budget_ms"]),
            int(row["cpsat_horizon"]),
            str(row["policy"]),
        ),
    )
    return {
        "selected_policy": str(selected["policy"]),
        "cpsat_horizon": int(selected["cpsat_horizon"]),
        "solver_budget_ms": float(selected["solver_budget_ms"]),
        "solver_seconds": float(selected["solver_budget_ms"]) / 1_000.0,
        "weighted_tardiness_mean": float(selected["weighted_tardiness_mean"]),
        "mean_decision_time_ms_mean": float(selected["mean_decision_time_ms_mean"]),
        "solver_fallback_rate_mean": float(selected["solver_fallback_rate_mean"]),
        "best_reliable_pareto_weighted_tardiness_mean": best_wtt,
        "quality_threshold_weighted_tardiness": threshold,
        "quality_tolerance_pct": quality_tolerance_pct,
        "max_fallback_rate_pct": max_fallback_rate_pct,
        "pareto_configurations": len(pareto),
        "reliable_pareto_configurations": len(reliable),
        "acceptable_pareto_configurations": len(acceptable),
    }


def build_freeze_manifest(
    selected: dict[str, Any],
    *,
    seed_start: int,
    seed_count: int,
    raw_input: Path,
    summary_input: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 5,
        "controller": "FJSP_ROLLING_HORIZON_CP_SAT",
        "status": "validation_selected",
        "selection_rule": (
            "Pareto WTT/latency, fallback <= declared limit, WTT within declared tolerance "
            "of best reliable Pareto point, then minimum measured latency."
        ),
        "validation_seed_start": seed_start,
        "validation_seed_count": seed_count,
        "validation_seed_end": seed_start + seed_count - 1,
        "raw_results": str(raw_input),
        "summary_results": str(summary_input),
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        **selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the Phase-5 FJSP CP-SAT operating point.")
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--summary-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seed-start", type=int, default=41000)
    parser.add_argument("--validation-seeds", type=int, default=30)
    parser.add_argument("--quality-tolerance-pct", type=float, default=2.0)
    parser.add_argument("--max-fallback-rate-pct", type=float, default=1.0)
    args = parser.parse_args()
    raw = read_csv(args.raw_input)
    summary = read_csv(args.summary_input)
    try:
        validate_validation_grid(
            raw,
            seed_start=args.validation_seed_start,
            seed_count=args.validation_seeds,
        )
        selected = select_operating_point(
            summary,
            quality_tolerance_pct=args.quality_tolerance_pct,
            max_fallback_rate_pct=args.max_fallback_rate_pct,
        )
    except ValueError as exc:
        parser.error(str(exc))
    manifest = build_freeze_manifest(
        selected,
        seed_start=args.validation_seed_start,
        seed_count=args.validation_seeds,
        raw_input=args.raw_input,
        summary_input=args.summary_input,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Selected FJSP CP-SAT: H={manifest['cpsat_horizon']} "
        f"budget={manifest['solver_budget_ms']:.1f} ms "
        f"WTT={manifest['weighted_tardiness_mean']:.3f} "
        f"latency={manifest['mean_decision_time_ms_mean']:.3f} ms "
        f"fallback={100.0 * manifest['solver_fallback_rate_mean']:.2f}%"
    )


if __name__ == "__main__":
    main()
