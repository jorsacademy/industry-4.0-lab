from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

VALIDATION_SEED_MIN = 10_000
FINAL_TEST_SEED_MIN = 20_000
STRESS_TEST_SEED_MIN = 30_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def validate_design(design: dict[str, Any]) -> dict[str, Any]:
    if design.get("algorithm") != "PPO":
        raise ValueError("campaign algorithm must be PPO")

    training_seeds = [int(seed) for seed in design.get("training_seeds", [])]
    if len(training_seeds) < 3:
        raise ValueError("campaign requires at least three independent training seeds")
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training seeds must be unique")
    if any(seed < 0 or seed >= VALIDATION_SEED_MIN for seed in training_seeds):
        raise ValueError("training seeds must remain below validation seed 10000")

    validation_start = int(design.get("validation_seed_start", -1))
    validation_count = int(design.get("validation_seed_count", 0))
    if validation_count <= 0:
        raise ValueError("validation seed count must be positive")
    validation_end_exclusive = validation_start + validation_count
    if validation_start < VALIDATION_SEED_MIN or validation_end_exclusive > FINAL_TEST_SEED_MIN:
        raise ValueError("validation seeds must remain in [10000, 20000)")

    if int(design.get("final_test_seed_start", -1)) < FINAL_TEST_SEED_MIN:
        raise ValueError("final-test seeds must start at or above 20000")
    if int(design.get("stress_test_seed_start", -1)) < STRESS_TEST_SEED_MIN:
        raise ValueError("stress-test seeds must start at or above 30000")

    training_config = design.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("training_config must be an object")
    if int(training_config.get("total_timesteps", 0)) <= 0:
        raise ValueError("total_timesteps must be positive")

    return {
        **design,
        "training_seeds": training_seeds,
        "validation_seed_start": validation_start,
        "validation_seed_count": validation_count,
        "final_test_seed_start": int(design["final_test_seed_start"]),
        "stress_test_seed_start": int(design["stress_test_seed_start"]),
    }


def _require_training_config_match(
    expected: dict[str, Any], observed: dict[str, Any], training_seed: int
) -> None:
    for key, expected_value in expected.items():
        if key == "seed":
            continue
        if key not in observed:
            raise ValueError(f"training manifest is missing training_config.{key}")
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not math.isclose(
                float(observed_value), expected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"training_config mismatch for {key}: {observed_value} != {expected_value}"
                )
        elif observed_value != expected_value:
            raise ValueError(
                f"training_config mismatch for {key}: {observed_value} != {expected_value}"
            )
    if int(observed.get("seed", -1)) != training_seed:
        raise ValueError("training manifest seed does not match campaign member")


def validate_member(
    design: dict[str, Any],
    artifacts_root: Path,
    training_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    member_dir = artifacts_root / f"seed_{training_seed}"
    runs_path = member_dir / "validation_runs.csv"
    summary_path = member_dir / "validation_summary.csv"
    manifest_path = member_dir / "training_manifest.json"
    model_path = member_dir / "ppo_dispatcher.zip"
    required = (runs_path, summary_path, manifest_path, model_path)
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty campaign artifact: {path}")

    raw_rows = read_csv(runs_path)
    ppo_rows = [row for row in raw_rows if row.get("policy") == "PPO"]
    expected_seeds = list(
        range(
            int(design["validation_seed_start"]),
            int(design["validation_seed_start"]) + int(design["validation_seed_count"]),
        )
    )
    observed_seeds = sorted(int(row["seed"]) for row in ppo_rows)
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"PPO validation seeds for training seed {training_seed} are incomplete or leaked"
        )

    summary_rows = read_csv(summary_path)
    ppo_summary = [row for row in summary_rows if row.get("policy") == "PPO"]
    if len(ppo_summary) != 1:
        raise ValueError(f"expected exactly one PPO summary row for training seed {training_seed}")
    summary = ppo_summary[0]

    training_manifest = read_json(manifest_path)
    if training_manifest.get("algorithm") != "PPO":
        raise ValueError("training manifest algorithm must be PPO")
    if int(training_manifest.get("training_seed", -1)) != training_seed:
        raise ValueError("training manifest training_seed mismatch")
    observed_training = training_manifest.get("training_config")
    if not isinstance(observed_training, dict):
        raise ValueError("training manifest training_config must be an object")
    _require_training_config_match(design["training_config"], observed_training, training_seed)

    fields = {
        "training_seed": training_seed,
        "n_validation_seeds": len(ppo_rows),
        "weighted_tardiness_mean": _finite_float(
            summary["weighted_tardiness_mean"], "weighted_tardiness_mean"
        ),
        "weighted_tardiness_ci_low": _finite_float(
            summary["weighted_tardiness_ci_low"], "weighted_tardiness_ci_low"
        ),
        "weighted_tardiness_ci_high": _finite_float(
            summary["weighted_tardiness_ci_high"], "weighted_tardiness_ci_high"
        ),
        "mean_decision_time_ms_mean": _finite_float(
            summary["mean_decision_time_ms_mean"], "mean_decision_time_ms_mean"
        ),
        "mean_waiting_time_mean": _finite_float(
            summary["mean_waiting_time_mean"], "mean_waiting_time_mean"
        ),
        "on_time_rate_mean": _finite_float(summary["on_time_rate_mean"], "on_time_rate_mean"),
        "makespan_mean": _finite_float(summary["makespan_mean"], "makespan_mean"),
        "training_seconds": _finite_float(
            training_manifest.get("training_seconds", 0.0), "training_seconds"
        ),
        "model_sha256": sha256_file(model_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "model_path": str(model_path),
        "training_manifest_path": str(manifest_path),
    }

    combined_rows: list[dict[str, Any]] = []
    for row in ppo_rows:
        combined_rows.append({"training_seed": training_seed, **row})
    return fields, combined_rows


def select_representative_member(member_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(member_rows) < 3:
        raise ValueError("at least three completed training members are required")
    wtt_values = [float(row["weighted_tardiness_mean"]) for row in member_rows]
    median_wtt = float(median(wtt_values))
    return min(
        member_rows,
        key=lambda row: (
            abs(float(row["weighted_tardiness_mean"]) - median_wtt),
            float(row["mean_decision_time_ms_mean"]),
            int(row["training_seed"]),
        ),
    )


def aggregate_member_metrics(member_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(member_rows) < 2:
        raise ValueError("at least two training members are required for aggregate dispersion")
    wtt = [float(row["weighted_tardiness_mean"]) for row in member_rows]
    latency = [float(row["mean_decision_time_ms_mean"]) for row in member_rows]
    return {
        "n_training_seeds": len(member_rows),
        "weighted_tardiness_training_mean": float(mean(wtt)),
        "weighted_tardiness_training_median": float(median(wtt)),
        "weighted_tardiness_training_std": float(stdev(wtt)),
        "weighted_tardiness_training_min": float(min(wtt)),
        "weighted_tardiness_training_max": float(max(wtt)),
        "mean_decision_time_ms_training_mean": float(mean(latency)),
    }


def build_campaign_manifest(
    design: dict[str, Any],
    member_rows: list[dict[str, Any]],
    representative: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "validation_complete_representative_frozen",
        "algorithm": "PPO",
        "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "selection_role": (
            "The representative model is for deployment/demo continuity only. Final scientific "
            "claims must aggregate all declared training seeds and must not cherry-pick this model."
        ),
        "representative_model_rule": (
            "Select the completed training seed whose validation weighted-tardiness mean is closest "
            "to the median across training seeds; break ties by lower decision latency, then seed."
        ),
        "representative_training_seed": int(representative["training_seed"]),
        "representative_model_sha256": representative["model_sha256"],
        "representative_validation_weighted_tardiness_mean": float(
            representative["weighted_tardiness_mean"]
        ),
        "training_design": design,
        "aggregate_validation": aggregate_member_metrics(member_rows),
        "members": member_rows,
        "final_test_used_for_selection": False,
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:  # pragma: no cover - CLI exercised in GitHub Actions
    parser = argparse.ArgumentParser(
        description="Validate and aggregate a multi-training-seed PPO validation campaign."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    try:
        design = validate_design(read_json(args.config))
        member_rows: list[dict[str, Any]] = []
        combined_rows: list[dict[str, Any]] = []
        for seed in design["training_seeds"]:
            member, rows = validate_member(design, args.artifacts_root, int(seed))
            member_rows.append(member)
            combined_rows.extend(rows)
        representative = select_representative_member(member_rows)
        campaign_manifest = build_campaign_manifest(design, member_rows, representative)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    write_csv(combined_rows, args.runs_output)
    write_csv(member_rows, args.summary_output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(campaign_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Representative PPO training seed: "
        f"{campaign_manifest['representative_training_seed']} "
        f"(median-role validation WTT="
        f"{campaign_manifest['representative_validation_weighted_tardiness_mean']:.3f})"
    )


if __name__ == "__main__":
    main()
