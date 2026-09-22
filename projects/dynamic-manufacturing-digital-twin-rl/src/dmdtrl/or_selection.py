from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

VALIDATION_SEED_MIN = 10_000
VALIDATION_SEED_MAX_EXCLUSIVE = 20_000
REQUIRED_SUMMARY_FIELDS = {
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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def validate_validation_seeds(
    raw_rows: list[dict[str, Any]],
    *,
    seed_start: int,
    seed_count: int,
) -> list[int]:
    if seed_count <= 0:
        raise ValueError("validation seed count must be positive")
    if seed_start < VALIDATION_SEED_MIN:
        raise ValueError("validation seeds must start at or above 10000")
    if seed_start + seed_count > VALIDATION_SEED_MAX_EXCLUSIVE:
        raise ValueError("validation seeds must remain below final-test seed 20000")
    if not raw_rows:
        raise ValueError("raw sensitivity results are empty")

    expected = list(range(seed_start, seed_start + seed_count))
    expected_set = set(expected)
    by_policy: dict[str, set[int]] = defaultdict(set)
    for row in raw_rows:
        by_policy[str(row["policy"])].add(int(row["seed"]))
    if len(by_policy) < 2:
        raise ValueError("raw sensitivity results must contain at least two configurations")

    for policy, observed in sorted(by_policy.items()):
        if observed != expected_set:
            raise ValueError(
                f"raw sensitivity seeds for {policy!r} do not match declared validation range: "
                f"expected {expected[0]}..{expected[-1]}, observed {sorted(observed)}"
            )
    return expected


def select_operating_point(
    summary_rows: list[dict[str, Any]],
    *,
    quality_tolerance_pct: float = 2.0,
    max_fallback_rate_pct: float = 1.0,
) -> dict[str, Any]:
    if quality_tolerance_pct < 0:
        raise ValueError("quality_tolerance_pct must be non-negative")
    if not 0.0 <= max_fallback_rate_pct <= 100.0:
        raise ValueError("max_fallback_rate_pct must be between 0 and 100")
    if not summary_rows:
        raise ValueError("sensitivity summary is empty")

    parsed: list[dict[str, Any]] = []
    for row in summary_rows:
        missing = REQUIRED_SUMMARY_FIELDS - set(row)
        if missing:
            raise ValueError(f"sensitivity summary is missing fields: {sorted(missing)}")
        weighted_tardiness = float(row["weighted_tardiness_mean"])
        latency = float(row["mean_decision_time_ms_mean"])
        fallback_rate = float(row["solver_fallback_rate_mean"])
        solver_budget_ms = float(row["solver_budget_ms"])
        horizon = int(float(row["cpsat_horizon"]))
        numeric_values = (weighted_tardiness, latency, fallback_rate, solver_budget_ms)
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("sensitivity summary contains a non-finite numeric value")
        if (
            weighted_tardiness < 0
            or latency < 0
            or not 0.0 <= fallback_rate <= 1.0
            or solver_budget_ms <= 0
            or horizon <= 0
        ):
            raise ValueError("sensitivity summary contains an invalid operational value")
        parsed.append(
            {
                **row,
                "weighted_tardiness_mean": weighted_tardiness,
                "mean_decision_time_ms_mean": latency,
                "solver_fallback_rate_mean": fallback_rate,
                "cpsat_horizon": horizon,
                "solver_budget_ms": solver_budget_ms,
                "pareto_optimal": _as_bool(row["pareto_optimal"]),
            }
        )

    pareto = [row for row in parsed if row["pareto_optimal"]]
    if not pareto:
        raise ValueError("sensitivity summary has no Pareto-optimal configuration")

    max_fallback_rate = max_fallback_rate_pct / 100.0
    reliable_pareto = [
        row for row in pareto if float(row["solver_fallback_rate_mean"]) <= max_fallback_rate + 1e-12
    ]
    if not reliable_pareto:
        raise ValueError(
            "no Pareto-optimal configuration satisfies the declared solver fallback-rate limit"
        )

    best_wtt = min(float(row["weighted_tardiness_mean"]) for row in reliable_pareto)
    threshold = best_wtt * (1.0 + quality_tolerance_pct / 100.0)
    acceptable = [
        row
        for row in reliable_pareto
        if float(row["weighted_tardiness_mean"]) <= threshold + 1e-12
    ]
    if not acceptable:  # pragma: no cover - best WTT is always acceptable
        raise RuntimeError("selection rule produced no acceptable configuration")

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
        "quality_tolerance_pct": float(quality_tolerance_pct),
        "max_fallback_rate_pct": float(max_fallback_rate_pct),
        "acceptable_pareto_configurations": len(acceptable),
        "reliable_pareto_configurations": len(reliable_pareto),
        "pareto_configurations": len(pareto),
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
        "controller": "CP_SAT_RH",
        "status": "validation_selected",
        "selection_rule": (
            "Restrict to Pareto-optimal configurations satisfying the declared maximum "
            "solver fallback rate. Within the declared WTT tolerance of the best reliable "
            "Pareto point, select the lowest measured mean decision latency; break ties by "
            "WTT, fallback rate, solver budget, horizon, then policy name."
        ),
        "validation_seed_start": seed_start,
        "validation_seed_count": seed_count,
        "validation_seed_end": seed_start + seed_count - 1,
        "raw_results": str(raw_input),
        "summary_results": str(summary_input),
        "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
        **selected,
    }


def main() -> None:  # pragma: no cover - CLI exercised by workflow smoke/validation
    parser = argparse.ArgumentParser(
        description="Select and freeze a CP-SAT operating point from validation sensitivity results."
    )
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--summary-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-tolerance-pct", type=float, default=2.0)
    parser.add_argument("--max-fallback-rate-pct", type=float, default=1.0)
    parser.add_argument("--validation-seed-start", type=int, default=10_000)
    parser.add_argument("--validation-seeds", type=int, default=30)
    args = parser.parse_args()

    raw_rows = read_csv(args.raw_input)
    summary_rows = read_csv(args.summary_input)
    try:
        validate_validation_seeds(
            raw_rows,
            seed_start=args.validation_seed_start,
            seed_count=args.validation_seeds,
        )
        selected = select_operating_point(
            summary_rows,
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
        "Selected CP-SAT operating point: "
        f"H={manifest['cpsat_horizon']} "
        f"budget={manifest['solver_budget_ms']:.1f} ms "
        f"validation WTT={manifest['weighted_tardiness_mean']:.3f} "
        f"decision={manifest['mean_decision_time_ms_mean']:.3f} ms "
        f"fallback={100.0 * manifest['solver_fallback_rate_mean']:.2f}%"
    )
    print(f"Freeze manifest saved to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
