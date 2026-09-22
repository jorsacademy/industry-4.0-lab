from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

from dmdtrl.statistics import bootstrap_mean_ci, paired_estimate

TRAINING_SEED_MAX_EXCLUSIVE = 40_000
OR_VALIDATION_SEED_MIN = 41_000
PPO_VALIDATION_SEED_MIN = 41_100
FINAL_SEED_MIN = 42_000
BASELINE_POLICIES = (
    "EARLIEST_DUE_DATE",
    "ROLLING_HORIZON_CPSAT",
    "SHORTEST_PROCESSING",
)


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
    if int(design.get("phase", -1)) != 5:
        raise ValueError("campaign phase must be 5")
    if design.get("algorithm") != "MaskablePPO":
        raise ValueError("campaign algorithm must be MaskablePPO")

    training_seeds = [int(seed) for seed in design.get("training_seeds", [])]
    if len(training_seeds) < 3:
        raise ValueError("campaign requires at least three independent training seeds")
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training seeds must be unique")
    if any(seed < 0 or seed >= TRAINING_SEED_MAX_EXCLUSIVE for seed in training_seeds):
        raise ValueError("training seeds must remain below Phase-5 evaluation seed 40000")

    validation_start = int(design.get("validation_seed_start", -1))
    validation_count = int(design.get("validation_seed_count", 0))
    if validation_count <= 0:
        raise ValueError("validation_seed_count must be positive")
    validation_end = validation_start + validation_count - 1
    if validation_start < PPO_VALIDATION_SEED_MIN or validation_end >= FINAL_SEED_MIN:
        raise ValueError("PPO validation seeds must remain in [41100, 42000)")
    if int(design.get("validation_seed_end", validation_end)) != validation_end:
        raise ValueError("validation_seed_end does not match start/count")

    or_start = int(design.get("or_tuning_seed_start", -1))
    or_count = int(design.get("or_tuning_seed_count", 0))
    if or_count <= 0:
        raise ValueError("or_tuning_seed_count must be positive")
    or_end = or_start + or_count - 1
    if or_start < OR_VALIDATION_SEED_MIN or or_end >= validation_start:
        raise ValueError("OR tuning seeds must precede and remain disjoint from PPO validation")
    if int(design.get("or_tuning_seed_end", or_end)) != or_end:
        raise ValueError("or_tuning_seed_end does not match start/count")

    final_start = int(design.get("final_test_seed_start", -1))
    final_count = int(design.get("final_test_seed_count", 0))
    if final_start < FINAL_SEED_MIN or final_count <= 0:
        raise ValueError("final-test seeds must start at 42000 or later with positive count")
    final_end = final_start + final_count - 1
    if int(design.get("final_test_seed_end", final_end)) != final_end:
        raise ValueError("final_test_seed_end does not match start/count")

    training_config = design.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("training_config must be an object")
    required_positive = (
        "total_timesteps",
        "learning_rate",
        "n_steps",
        "batch_size",
        "gamma",
        "hidden_units",
    )
    for field in required_positive:
        if float(training_config.get(field, 0.0)) <= 0.0:
            raise ValueError(f"training_config.{field} must be positive")
    if int(training_config["n_steps"]) % int(training_config["batch_size"]) != 0:
        raise ValueError("training_config.n_steps must be divisible by batch_size")
    if not 0.0 <= float(training_config.get("gae_lambda", -1.0)) <= 1.0:
        raise ValueError("training_config.gae_lambda must be in [0, 1]")
    if float(training_config.get("ent_coef", -1.0)) < 0.0:
        raise ValueError("training_config.ent_coef must be non-negative")

    environment = design.get("environment_config")
    if not isinstance(environment, dict) or not isinstance(environment.get("generator"), dict):
        raise ValueError("environment_config.generator must be an object")
    if not str(design.get("frozen_cpsat_config_path", "")):
        raise ValueError("frozen_cpsat_config_path is required")

    return {
        **design,
        "training_seeds": training_seeds,
        "validation_seed_start": validation_start,
        "validation_seed_count": validation_count,
        "validation_seed_end": validation_end,
        "or_tuning_seed_start": or_start,
        "or_tuning_seed_count": or_count,
        "or_tuning_seed_end": or_end,
        "final_test_seed_start": final_start,
        "final_test_seed_count": final_count,
        "final_test_seed_end": final_end,
    }


def validate_cpsat_freeze(freeze: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    if freeze.get("status") != "validation_selected_frozen":
        raise ValueError("CP-SAT freeze must have validation_selected_frozen status")
    if freeze.get("controller") != "FJSP_ROLLING_HORIZON_CP_SAT":
        raise ValueError("unexpected frozen CP-SAT controller")
    if freeze.get("selected_policy") != design.get("frozen_cpsat_policy"):
        raise ValueError("campaign frozen_cpsat_policy does not match CP-SAT freeze")

    grid = freeze.get("selection_grid")
    config = freeze.get("cpsat_config")
    metrics = freeze.get("validation_metrics")
    boundary = freeze.get("selection_data_boundary")
    if not all(isinstance(value, dict) for value in (grid, config, metrics, boundary)):
        raise ValueError("CP-SAT freeze is missing required structured fields")

    if int(grid["validation_seed_start"]) != int(design["or_tuning_seed_start"]):
        raise ValueError("CP-SAT tuning seed start does not match campaign design")
    if int(grid["validation_seed_count"]) != int(design["or_tuning_seed_count"]):
        raise ValueError("CP-SAT tuning seed count does not match campaign design")
    if int(grid["validation_seed_end"]) >= int(design["validation_seed_start"]):
        raise ValueError("CP-SAT tuning data overlaps PPO validation data")
    if float(metrics["solver_fallback_rate_mean"]) > 0.01 + 1e-12:
        raise ValueError("frozen CP-SAT operating point violates the declared fallback limit")
    if bool(boundary.get("final_test_used_for_selection")):
        raise ValueError("CP-SAT freeze must not use final-test data for selection")

    if int(config.get("job_horizon", 0)) <= 0 or float(config.get("solver_seconds", 0.0)) <= 0.0:
        raise ValueError("frozen CP-SAT operating point is invalid")
    if int(config.get("num_search_workers", 0)) != 1:
        raise ValueError("frozen CP-SAT must retain one search worker")
    return dict(config)


def _expected_validation_seeds(design: dict[str, Any]) -> list[int]:
    return list(
        range(
            int(design["validation_seed_start"]),
            int(design["validation_seed_start"]) + int(design["validation_seed_count"]),
        )
    )


def validate_baseline_panel(
    design: dict[str, Any],
    baseline_root: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[int, str],
]:
    runs_path = baseline_root / "baseline_runs.csv"
    summary_path = baseline_root / "baseline_summary.csv"
    for path in (runs_path, summary_path):
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty baseline artifact: {path}")

    rows = read_csv(runs_path)
    expected_seeds = _expected_validation_seeds(design)
    expected_set = set(expected_seeds)
    by_policy: dict[str, set[int]] = defaultdict(set)
    fingerprints: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        policy = str(row.get("policy"))
        if policy not in BASELINE_POLICIES:
            raise ValueError(f"unexpected baseline policy {policy!r}")
        seed = int(row["seed"])
        if row.get("seed_regime") != "validation":
            raise ValueError("baseline row escaped the Phase-5 validation regime")
        by_policy[policy].add(seed)
        fingerprints[seed].add(str(row["instance_sha256"]))

    if set(by_policy) != set(BASELINE_POLICIES):
        raise ValueError("baseline panel does not contain the exact declared policy set")
    for policy, observed in by_policy.items():
        if observed != expected_set:
            raise ValueError(f"baseline seed set is incomplete for {policy}")
    if len(rows) != len(BASELINE_POLICIES) * len(expected_seeds):
        raise ValueError("baseline panel contains duplicate or missing rows")
    if set(fingerprints) != expected_set or any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError("baseline policies did not use one canonical instance per seed")

    summary = read_csv(summary_path)
    if {row.get("policy") for row in summary} != set(BASELINE_POLICIES):
        raise ValueError("baseline summary does not match the declared policy set")
    fingerprint_by_seed = {seed: next(iter(values)) for seed, values in fingerprints.items()}
    return rows, summary, fingerprint_by_seed


def _require_training_config_match(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if key not in observed:
            raise ValueError(f"training manifest is missing training_config.{key}")
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not math.isclose(
                float(observed_value), float(expected_value), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"training_config mismatch for {key}: {observed_value} != {expected_value}"
                )
        elif observed_value != expected_value:
            raise ValueError(
                f"training_config mismatch for {key}: {observed_value} != {expected_value}"
            )


def validate_member(
    design: dict[str, Any],
    members_root: Path,
    training_seed: int,
    expected_fingerprints: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    member_dir = members_root / f"seed_{training_seed}"
    model_path = member_dir / "fjsp_maskable_ppo.zip"
    manifest_path = member_dir / "training_manifest.json"
    runs_path = member_dir / "ppo_validation_runs.csv"
    summary_path = member_dir / "ppo_validation_summary.csv"
    for path in (model_path, manifest_path, runs_path, summary_path):
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty training-member artifact: {path}")

    rows = read_csv(runs_path)
    expected_seeds = _expected_validation_seeds(design)
    observed_seeds = sorted(int(row["seed"]) for row in rows)
    if observed_seeds != expected_seeds:
        raise ValueError(f"validation seeds are incomplete for training seed {training_seed}")
    for row in rows:
        seed = int(row["seed"])
        if row.get("policy") != "MASKABLE_PPO":
            raise ValueError("member validation artifact contains a non-PPO policy")
        if int(row.get("training_seed", -1)) != training_seed:
            raise ValueError("member validation row training_seed mismatch")
        if row.get("seed_regime") != "validation":
            raise ValueError("member validation row escaped the validation seed regime")
        if row.get("instance_sha256") != expected_fingerprints[seed]:
            raise ValueError("PPO and baseline panels used different common-seed instances")

    summary_rows = read_csv(summary_path)
    if len(summary_rows) != 1 or summary_rows[0].get("policy") != "MASKABLE_PPO":
        raise ValueError("member summary must contain exactly one Maskable PPO row")
    summary = summary_rows[0]
    if int(summary.get("training_seed", -1)) != training_seed:
        raise ValueError("member summary training_seed mismatch")

    manifest = read_json(manifest_path)
    if manifest.get("algorithm") != "MaskablePPO":
        raise ValueError("training manifest algorithm must be MaskablePPO")
    if int(manifest.get("training_seed", -1)) != training_seed:
        raise ValueError("training manifest seed mismatch")
    observed_training = manifest.get("training_config")
    observed_environment = manifest.get("environment_config")
    if not isinstance(observed_training, dict) or not isinstance(observed_environment, dict):
        raise ValueError("training manifest is missing training/environment configuration")
    _require_training_config_match(design["training_config"], observed_training)
    if observed_environment != design["environment_config"]:
        raise ValueError("training environment configuration drifted from campaign design")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("sb3_contrib") in {None, "not-installed"}:
        raise ValueError("training manifest does not prove a real sb3-contrib runtime")

    member = {
        "training_seed": training_seed,
        "n_validation_seeds": len(rows),
        "weighted_tardiness_mean": _finite_float(
            summary["weighted_tardiness_mean"], "weighted_tardiness_mean"
        ),
        "weighted_tardiness_std": _finite_float(
            summary["weighted_tardiness_std"], "weighted_tardiness_std"
        ),
        "weighted_tardiness_ci_low": _finite_float(
            summary["weighted_tardiness_ci_low"], "weighted_tardiness_ci_low"
        ),
        "weighted_tardiness_ci_high": _finite_float(
            summary["weighted_tardiness_ci_high"], "weighted_tardiness_ci_high"
        ),
        "makespan_mean": _finite_float(summary["makespan_mean"], "makespan_mean"),
        "mean_flow_time": _finite_float(summary["mean_flow_time"], "mean_flow_time"),
        "mean_decision_time_ms": _finite_float(
            summary["mean_decision_time_ms"], "mean_decision_time_ms"
        ),
        "training_seconds": _finite_float(manifest["training_seconds"], "training_seconds"),
        "model_sha256": sha256_file(model_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "model_path": str(model_path),
        "training_manifest_path": str(manifest_path),
    }
    return member, rows


def compare_member_to_baselines(
    member_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    *,
    training_seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> list[dict[str, Any]]:
    ppo_by_seed = {int(row["seed"]): row for row in member_rows}
    comparisons: list[dict[str, Any]] = []
    for index, baseline in enumerate(BASELINE_POLICIES):
        baseline_by_seed = {
            int(row["seed"]): row for row in baseline_rows if row["policy"] == baseline
        }
        if set(baseline_by_seed) != set(ppo_by_seed):
            raise ValueError("paired baseline/PPO seed sets are not identical")
        seeds = sorted(ppo_by_seed)
        differences = np.asarray(
            [
                float(baseline_by_seed[seed]["weighted_tardiness"])
                - float(ppo_by_seed[seed]["weighted_tardiness"])
                for seed in seeds
            ],
            dtype=float,
        )
        estimate = paired_estimate(
            differences,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=91_000 + training_seed + index * 100,
        )
        baseline_mean = float(
            np.mean([float(baseline_by_seed[seed]["weighted_tardiness"]) for seed in seeds])
        )
        comparisons.append(
            {
                "training_seed": training_seed,
                "candidate": "MASKABLE_PPO",
                "baseline": baseline,
                "metric": "weighted_tardiness",
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
                "n_instance_pairs": estimate.n_pairs,
            }
        )
    return comparisons


def aggregate_member_metrics(member_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(member_rows) < 2:
        raise ValueError("at least two training members are required for aggregate dispersion")
    wtt = [float(row["weighted_tardiness_mean"]) for row in member_rows]
    latency = [float(row["mean_decision_time_ms"]) for row in member_rows]
    training_seconds = [float(row["training_seconds"]) for row in member_rows]
    return {
        "n_training_seeds": len(member_rows),
        "weighted_tardiness_training_mean": float(mean(wtt)),
        "weighted_tardiness_training_median": float(median(wtt)),
        "weighted_tardiness_training_std": float(stdev(wtt)),
        "weighted_tardiness_training_min": float(min(wtt)),
        "weighted_tardiness_training_max": float(max(wtt)),
        "mean_decision_time_ms_training_mean": float(mean(latency)),
        "training_seconds_mean": float(mean(training_seconds)),
        "training_seconds_total": float(sum(training_seconds)),
    }


def aggregate_comparisons(
    comparison_rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, baseline in enumerate(BASELINE_POLICIES):
        rows = [row for row in comparison_rows if row["baseline"] == baseline]
        improvements = [float(row["mean_improvement"]) for row in rows]
        if len(improvements) < 2:
            raise ValueError("aggregate comparison requires multiple training seeds")
        estimate = bootstrap_mean_ci(
            improvements,
            n_bootstrap=n_bootstrap,
            seed=97_000 + index,
        )
        output.append(
            {
                "candidate": "MASKABLE_PPO",
                "baseline": baseline,
                "metric": "training_seed_mean_paired_weighted_tardiness_improvement",
                "n_training_seeds": len(improvements),
                "mean_improvement_across_training_seeds": estimate.mean,
                "std_improvement_across_training_seeds": estimate.std,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "median_improvement": float(median(improvements)),
                "min_improvement": float(min(improvements)),
                "max_improvement": float(max(improvements)),
                "training_seed_win_fraction": float(
                    np.mean(np.asarray(improvements, dtype=float) > 0.0)
                ),
                "inference_unit": "training_seed",
            }
        )
    return output


def select_representative_member(member_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(member_rows) < 3:
        raise ValueError("at least three completed training members are required")
    median_wtt = float(median(float(row["weighted_tardiness_mean"]) for row in member_rows))
    return min(
        member_rows,
        key=lambda row: (
            abs(float(row["weighted_tardiness_mean"]) - median_wtt),
            float(row["mean_decision_time_ms"]),
            int(row["training_seed"]),
        ),
    )


def build_campaign_manifest(
    design: dict[str, Any],
    freeze: dict[str, Any],
    member_rows: list[dict[str, Any]],
    baseline_summary: list[dict[str, str]],
    aggregate_rows: list[dict[str, Any]],
    representative: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 5,
        "status": "validation_complete_representative_frozen",
        "algorithm": "MaskablePPO",
        "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "training_design": design,
        "frozen_cpsat": {
            "selected_policy": freeze["selected_policy"],
            "cpsat_config": freeze["cpsat_config"],
            "source_pull_request": freeze["source_pull_request"],
            "source_workflow_run_id": freeze["source_workflow_run_id"],
            "source_pull_request_merge_sha": freeze["source_pull_request_merge_sha"],
        },
        "aggregate_validation": aggregate_member_metrics(member_rows),
        "aggregate_training_seed_comparisons": aggregate_rows,
        "baseline_validation_summary": baseline_summary,
        "representative_model_rule": design["representative_model_rule"],
        "representative_model_role": "deployment_and_demo_only",
        "representative_training_seed": int(representative["training_seed"]),
        "representative_model_sha256": representative["model_sha256"],
        "representative_training_manifest_sha256": representative[
            "training_manifest_sha256"
        ],
        "representative_validation_weighted_tardiness_mean": float(
            representative["weighted_tardiness_mean"]
        ),
        "members": member_rows,
        "scientific_claim_rule": design["scientific_claim_rule"],
        "final_test_used_for_selection": False,
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:  # pragma: no cover - exercised by GitHub Actions
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the Phase-5 multi-seed Maskable PPO campaign."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpsat-freeze", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--members-root", type=Path, required=True)
    parser.add_argument("--members-output", type=Path, required=True)
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--comparisons-output", type=Path, required=True)
    parser.add_argument("--aggregate-comparisons-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    try:
        design = validate_design(read_json(args.config))
        freeze = read_json(args.cpsat_freeze)
        validate_cpsat_freeze(freeze, design)
        baseline_rows, baseline_summary, fingerprints = validate_baseline_panel(
            design, args.baseline_root
        )

        member_rows: list[dict[str, Any]] = []
        ppo_rows: list[dict[str, str]] = []
        comparison_rows: list[dict[str, Any]] = []
        for seed in design["training_seeds"]:
            member, rows = validate_member(
                design,
                args.members_root,
                int(seed),
                fingerprints,
            )
            member_rows.append(member)
            ppo_rows.extend(rows)
            comparison_rows.extend(
                compare_member_to_baselines(
                    rows,
                    baseline_rows,
                    training_seed=int(seed),
                    n_bootstrap=args.bootstrap,
                    n_permutations=args.permutations,
                )
            )
        aggregate_rows = aggregate_comparisons(
            comparison_rows,
            n_bootstrap=args.bootstrap,
        )
        representative = select_representative_member(member_rows)
        manifest = build_campaign_manifest(
            design,
            freeze,
            member_rows,
            baseline_summary,
            aggregate_rows,
            representative,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    combined_runs: list[dict[str, Any]] = [
        {"training_seed": "", **row} for row in baseline_rows
    ] + [dict(row) for row in ppo_rows]
    write_csv(member_rows, args.members_output)
    write_csv(combined_runs, args.runs_output)
    write_csv(comparison_rows, args.comparisons_output)
    write_csv(aggregate_rows, args.aggregate_comparisons_output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Representative Maskable PPO training seed: "
        f"{manifest['representative_training_seed']} "
        f"(median-role validation WTT="
        f"{manifest['representative_validation_weighted_tardiness_mean']:.3f})"
    )
    for row in aggregate_rows:
        print(
            f"PPO vs {row['baseline']}: "
            f"training-seed mean paired improvement="
            f"{float(row['mean_improvement_across_training_seeds']):.3f}, "
            f"seed-win={100.0 * float(row['training_seed_win_fraction']):.1f}%"
        )


if __name__ == "__main__":
    main()
