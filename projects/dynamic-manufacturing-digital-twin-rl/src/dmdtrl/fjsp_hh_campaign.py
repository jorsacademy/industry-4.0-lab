from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from dmdtrl.fjsp_env import FJSPEnvConfig
from dmdtrl.fjsp_evaluate import instance_fingerprint, summarize_panel
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_hh_evaluate import _run_cpsat, _run_operator
from dmdtrl.fjsp_hh_protocol import load_hh_validation_design
from dmdtrl.fjsp_hh_train import FJSPHyperHeuristicPPOConfig
from dmdtrl.fjsp_operators import OPERATOR_NAMES, FJSPOperator
from dmdtrl.fjsp_optimization import FJSPCPSATConfig
from dmdtrl.fjsp_ppo_campaign import (
    aggregate_member_metrics,
    read_csv,
    read_json,
    select_representative_member,
    sha256_file,
    write_csv,
)
from dmdtrl.statistics import bootstrap_mean_ci, paired_estimate

PPO_POLICY = "PPO_HYPER_HEURISTIC"
CPSAT_POLICY = "ROLLING_HORIZON_CPSAT"


def build_generator_config(design: dict[str, Any]) -> FJSPGeneratorConfig:
    generator = design["environment_config"]["generator"]
    config = FJSPGeneratorConfig(
        n_jobs=int(generator["n_jobs"]),
        n_machines=int(generator["n_machines"]),
        n_families=int(generator["n_families"]),
        operations_min=int(generator["operations_min"]),
        operations_max=int(generator["operations_max"]),
        eligible_machines_min=int(generator["eligible_machines_min"]),
        eligible_machines_max=int(generator["eligible_machines_max"]),
        mean_interarrival=float(generator["mean_interarrival"]),
        processing_min=float(generator["processing_min"]),
        processing_max=float(generator["processing_max"]),
        due_date_factor_min=float(generator["due_date_factor_min"]),
        due_date_factor_max=float(generator["due_date_factor_max"]),
    )
    config.validate()
    return config


def build_env_config(design: dict[str, Any]) -> FJSPEnvConfig:
    environment = design["environment_config"]
    config = FJSPEnvConfig(
        generator=build_generator_config(design),
        default_setup_time=float(environment["default_setup_time"]),
        operation_bonus=float(environment["operation_bonus"]),
        job_completion_bonus=float(environment["job_completion_bonus"]),
        waiting_weight=float(environment["waiting_weight"]),
        setup_weight=float(environment["setup_weight"]),
        tardiness_weight=float(environment["tardiness_weight"]),
    )
    config.validate()
    return config


def build_training_config(design: dict[str, Any], training_seed: int) -> FJSPHyperHeuristicPPOConfig:
    if training_seed not in design["training_seeds"]:
        raise ValueError("training seed is not part of the predeclared campaign")
    training = design["training_config"]
    config = FJSPHyperHeuristicPPOConfig(
        total_timesteps=int(training["total_timesteps"]),
        seed=int(training_seed),
        learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]),
        batch_size=int(training["batch_size"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        ent_coef=float(training["ent_coef"]),
        hidden_units=int(training["hidden_units"]),
        device=str(training["device"]),
        verbose=0,
    )
    config.validate()
    return config


def validate_cpsat_freeze(freeze: dict[str, Any], design: dict[str, Any]) -> FJSPCPSATConfig:
    if freeze.get("status") != "validation_selected_frozen":
        raise ValueError("CP-SAT freeze must be validation_selected_frozen")
    if freeze.get("controller") != "FJSP_ROLLING_HORIZON_CP_SAT":
        raise ValueError("unexpected CP-SAT controller")
    if freeze.get("selected_policy") != design.get("frozen_cpsat_policy"):
        raise ValueError("frozen CP-SAT policy does not match validation design")
    if freeze.get("selection_data_boundary", {}).get("final_test_used_for_selection") is not False:
        raise ValueError("frozen CP-SAT selection must not use final-test data")
    config = freeze.get("cpsat_config")
    if not isinstance(config, dict):
        raise ValueError("CP-SAT freeze is missing cpsat_config")
    if int(config.get("num_search_workers", 0)) != 1:
        raise ValueError("frozen CP-SAT must retain one search worker")
    policy = FJSPCPSATConfig(
        job_horizon=int(config["job_horizon"]),
        solver_seconds=float(config["solver_seconds"]),
        time_scale=int(config["time_scale"]),
        random_seed=int(config["random_seed"]),
        num_search_workers=int(config["num_search_workers"]),
    )
    policy.validate()
    return policy


def validation_seeds(design: dict[str, Any]) -> list[int]:
    start = int(design["validation_seed_start"])
    count = int(design["validation_seed_count"])
    end = int(design["validation_seed_end"])
    seeds = list(range(start, start + count))
    if not seeds or seeds[-1] != end:
        raise ValueError("validation seed range does not match the predeclared design")
    if start != 41_200 or end != 41_229:
        raise ValueError("hyper-heuristic validation must remain 41200-41229")
    if end >= int(design["final_test_seed_start"]):
        raise ValueError("validation seeds overlap the final-test embargo")
    return seeds


def baseline_policies(design: dict[str, Any]) -> tuple[str, ...]:
    operators = tuple(design["fixed_operator_baselines"])
    if operators != OPERATOR_NAMES:
        raise ValueError("fixed operator baseline set drifted from the action space")
    return (*operators, CPSAT_POLICY)


def evaluate_baseline_panel(
    design: dict[str, Any],
    freeze: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    generator = build_generator_config(design)
    env = build_env_config(design)
    cpsat = validate_cpsat_freeze(freeze, design)
    rows: list[dict[str, float | int | str]] = []
    for seed in validation_seeds(design):
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator)
        fingerprint = instance_fingerprint(instance)
        results = [
            _run_operator(instance, operator, setup_time=env.default_setup_time)
            for operator in FJSPOperator
        ]
        results.append(_run_cpsat(instance, setup_time=env.default_setup_time, config=cpsat))
        for result in results:
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "validation",
                    "instance_sha256": fingerprint,
                    **result,
                }
            )
    return rows


def validate_baseline_panel(
    design: dict[str, Any], baseline_root: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[int, str]]:
    runs_path = baseline_root / "baseline_runs.csv"
    summary_path = baseline_root / "baseline_summary.csv"
    for path in (runs_path, summary_path):
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty baseline artifact: {path}")
    rows = read_csv(runs_path)
    expected_seeds = set(validation_seeds(design))
    expected_policies = set(baseline_policies(design))
    by_policy: dict[str, set[int]] = defaultdict(set)
    fingerprints: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        policy = row["policy"]
        if policy not in expected_policies:
            raise ValueError(f"unexpected baseline policy {policy!r}")
        seed = int(row["seed"])
        if row.get("seed_regime") != "validation":
            raise ValueError("baseline row escaped the validation regime")
        by_policy[policy].add(seed)
        fingerprints[seed].add(row["instance_sha256"])
    if set(by_policy) != expected_policies:
        raise ValueError("baseline panel does not contain the exact frozen policy set")
    if any(seeds != expected_seeds for seeds in by_policy.values()):
        raise ValueError("baseline panel has incomplete validation seeds")
    if len(rows) != len(expected_policies) * len(expected_seeds):
        raise ValueError("baseline panel contains duplicate or missing rows")
    if set(fingerprints) != expected_seeds or any(len(values) != 1 for values in fingerprints.values()):
        raise ValueError("baseline policies did not share one canonical instance per seed")
    summary = read_csv(summary_path)
    if {row["policy"] for row in summary} != expected_policies:
        raise ValueError("baseline summary does not match the frozen policy set")
    return rows, summary, {seed: next(iter(values)) for seed, values in fingerprints.items()}


def _require_training_config_match(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if key not in observed:
            raise ValueError(f"training manifest is missing training_config.{key}")
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not math.isclose(float(observed_value), expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"training_config mismatch for {key}")
        elif observed_value != expected_value:
            raise ValueError(f"training_config mismatch for {key}")


def validate_member(
    design: dict[str, Any],
    members_root: Path,
    training_seed: int,
    expected_fingerprints: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    member_dir = members_root / f"seed_{training_seed}"
    model_path = member_dir / "fjsp_hyperheuristic_ppo.zip"
    manifest_path = member_dir / "training_manifest.json"
    runs_path = member_dir / "ppo_validation_runs.csv"
    summary_path = member_dir / "ppo_validation_summary.csv"
    for path in (model_path, manifest_path, runs_path, summary_path):
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty training-member artifact: {path}")

    rows = read_csv(runs_path)
    expected_seeds = validation_seeds(design)
    if sorted(int(row["seed"]) for row in rows) != expected_seeds:
        raise ValueError(f"validation seeds are incomplete for training seed {training_seed}")
    for row in rows:
        seed = int(row["seed"])
        if row.get("policy") != PPO_POLICY:
            raise ValueError("member validation artifact contains a non-PPO policy")
        if int(row.get("training_seed", -1)) != training_seed:
            raise ValueError("member validation row training_seed mismatch")
        if row.get("seed_regime") != "validation":
            raise ValueError("member validation row escaped the validation regime")
        if row.get("instance_sha256") != expected_fingerprints[seed]:
            raise ValueError("PPO and baseline panels used different common-seed instances")

    summary_rows = read_csv(summary_path)
    if len(summary_rows) != 1 or summary_rows[0].get("policy") != PPO_POLICY:
        raise ValueError("member summary must contain exactly one PPO hyper-heuristic row")
    summary = summary_rows[0]
    if int(summary.get("training_seed", -1)) != training_seed:
        raise ValueError("member summary training_seed mismatch")

    manifest = read_json(manifest_path)
    if manifest.get("algorithm") != "PPO" or manifest.get("controller") != "FJSP_HYPER_HEURISTIC":
        raise ValueError("training manifest algorithm/controller mismatch")
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
    if not isinstance(runtime, dict) or runtime.get("stable_baselines3") in {None, "not-installed"}:
        raise ValueError("training manifest does not prove a real Stable-Baselines3 runtime")

    def finite(field: str) -> float:
        value = float(summary[field])
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value

    member = {
        "training_seed": training_seed,
        "n_validation_seeds": len(rows),
        "weighted_tardiness_mean": finite("weighted_tardiness_mean"),
        "weighted_tardiness_std": finite("weighted_tardiness_std"),
        "weighted_tardiness_ci_low": finite("weighted_tardiness_ci_low"),
        "weighted_tardiness_ci_high": finite("weighted_tardiness_ci_high"),
        "makespan_mean": finite("makespan_mean"),
        "mean_flow_time": finite("mean_flow_time"),
        "mean_decision_time_ms": finite("mean_decision_time_ms"),
        "training_seconds": float(manifest["training_seconds"]),
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
    policies: tuple[str, ...],
    n_bootstrap: int,
    n_permutations: int,
) -> list[dict[str, Any]]:
    ppo_by_seed = {int(row["seed"]): row for row in member_rows}
    output: list[dict[str, Any]] = []
    for index, baseline in enumerate(policies):
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
            seed=101_000 + training_seed + index * 100,
        )
        baseline_mean = float(
            np.mean([float(baseline_by_seed[seed]["weighted_tardiness"]) for seed in seeds])
        )
        output.append(
            {
                "training_seed": training_seed,
                "candidate": PPO_POLICY,
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
    return output


def aggregate_comparisons(
    comparison_rows: list[dict[str, Any]], policies: tuple[str, ...], *, n_bootstrap: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, baseline in enumerate(policies):
        rows = [row for row in comparison_rows if row["baseline"] == baseline]
        improvements = [float(row["mean_improvement"]) for row in rows]
        if len(improvements) < 2:
            raise ValueError("aggregate comparison requires multiple training seeds")
        estimate = bootstrap_mean_ci(improvements, n_bootstrap=n_bootstrap, seed=107_000 + index)
        output.append(
            {
                "candidate": PPO_POLICY,
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
                "training_seed_win_fraction": float(np.mean(np.asarray(improvements) > 0.0)),
                "inference_unit": "training_seed",
            }
        )
    return output


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
        "status": "hyperheuristic_validation_complete_representative_frozen",
        "algorithm": "PPO",
        "controller": "FJSP_HYPER_HEURISTIC",
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
        "representative_training_manifest_sha256": representative["training_manifest_sha256"],
        "representative_validation_weighted_tardiness_mean": float(
            representative["weighted_tardiness_mean"]
        ),
        "members": member_rows,
        "scientific_claim_rule": design["scientific_claim_rule"],
        "final_test_used_for_selection": False,
    }


def run_aggregate(
    *,
    design: dict[str, Any],
    freeze: dict[str, Any],
    baseline_root: Path,
    members_root: Path,
    n_bootstrap: int,
    n_permutations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_cpsat_freeze(freeze, design)
    baseline_rows, baseline_summary, fingerprints = validate_baseline_panel(design, baseline_root)
    policies = baseline_policies(design)
    members: list[dict[str, Any]] = []
    ppo_rows: list[dict[str, str]] = []
    comparisons: list[dict[str, Any]] = []
    for seed in design["training_seeds"]:
        member, rows = validate_member(design, members_root, int(seed), fingerprints)
        members.append(member)
        ppo_rows.extend(rows)
        comparisons.extend(
            compare_member_to_baselines(
                rows,
                baseline_rows,
                training_seed=int(seed),
                policies=policies,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
            )
        )
    aggregate = aggregate_comparisons(comparisons, policies, n_bootstrap=n_bootstrap)
    representative = select_representative_member(members)
    manifest = build_campaign_manifest(
        design, freeze, members, baseline_summary, aggregate, representative
    )
    combined_runs: list[dict[str, Any]] = [
        {"training_seed": "", **row} for row in baseline_rows
    ] + [dict(row) for row in ppo_rows]
    return members, combined_runs, comparisons, manifest


def main() -> None:  # pragma: no cover - exercised by GitHub Actions
    parser = argparse.ArgumentParser(description="Run or aggregate FJSP hyper-heuristic validation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--config", type=Path, required=True)
    baseline.add_argument("--cpsat-freeze", type=Path, required=True)
    baseline.add_argument("--output-root", type=Path, required=True)
    baseline.add_argument("--bootstrap", type=int, default=2_000)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--cpsat-freeze", type=Path, required=True)
    aggregate.add_argument("--baseline-root", type=Path, required=True)
    aggregate.add_argument("--members-root", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--bootstrap", type=int, default=5_000)
    aggregate.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()

    design = load_hh_validation_design(args.config)
    freeze = read_json(args.cpsat_freeze)
    if args.command == "baseline":
        rows = evaluate_baseline_panel(design, freeze)
        summary = summarize_panel(rows, bootstrap=args.bootstrap)
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.output_root / "baseline_runs.csv")
        write_csv(summary, args.output_root / "baseline_summary.csv")
        return

    members, runs, comparisons, manifest = run_aggregate(
        design=design,
        freeze=freeze,
        baseline_root=args.baseline_root,
        members_root=args.members_root,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
    )
    aggregate_rows = manifest["aggregate_training_seed_comparisons"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(members, args.output_root / "fjsp_hh_training_seed_summary.csv")
    write_csv(runs, args.output_root / "fjsp_hh_validation_runs.csv")
    write_csv(comparisons, args.output_root / "fjsp_hh_member_comparisons.csv")
    write_csv(aggregate_rows, args.output_root / "fjsp_hh_training_seed_comparisons.csv")
    (args.output_root / "fjsp_hh_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
