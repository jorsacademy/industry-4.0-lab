from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dmdtrl.fjsp_env import FJSPEnvConfig
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_ppo_campaign import (
    aggregate_comparisons,
    compare_member_to_baselines,
    select_representative_member,
    validate_baseline_panel,
    validate_cpsat_freeze,
    validate_design,
    validate_member,
    write_csv,
)
from dmdtrl.fjsp_ppo_member import evaluate_member


def _design() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": 5,
        "algorithm": "MaskablePPO",
        "training_seeds": [7, 17, 27],
        "training_config": {
            "total_timesteps": 1000,
            "learning_rate": 0.0003,
            "n_steps": 64,
            "batch_size": 16,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "hidden_units": 32,
            "device": "cpu",
        },
        "environment_config": {
            "generator": {
                "n_jobs": 3,
                "n_machines": 2,
                "n_families": 2,
                "operations_min": 1,
                "operations_max": 2,
                "eligible_machines_min": 1,
                "eligible_machines_max": 2,
                "mean_interarrival": 1.0,
                "processing_min": 1.0,
                "processing_max": 3.0,
                "due_date_factor_min": 1.5,
                "due_date_factor_max": 2.5,
            },
            "default_setup_time": 0.5,
            "operation_bonus": 0.05,
            "job_completion_bonus": 1.0,
            "waiting_weight": 0.02,
            "setup_weight": 0.05,
            "tardiness_weight": 0.2,
        },
        "validation_seed_start": 41100,
        "validation_seed_count": 2,
        "validation_seed_end": 41101,
        "or_tuning_seed_start": 41000,
        "or_tuning_seed_count": 2,
        "or_tuning_seed_end": 41001,
        "final_test_seed_start": 42000,
        "final_test_seed_count": 10,
        "final_test_seed_end": 42009,
        "frozen_cpsat_config_path": "configs/fjsp_cpsat_validation_freeze.json",
        "frozen_cpsat_policy": "FJSP_CPSAT_H4_B100MS",
        "representative_model_rule": "median role",
        "scientific_claim_rule": "retain every training seed",
    }


def _freeze() -> dict[str, object]:
    return {
        "status": "validation_selected_frozen",
        "controller": "FJSP_ROLLING_HORIZON_CP_SAT",
        "selected_policy": "FJSP_CPSAT_H4_B100MS",
        "source_pull_request": 18,
        "source_workflow_run_id": 123,
        "source_pull_request_merge_sha": "abc",
        "selection_grid": {
            "validation_seed_start": 41000,
            "validation_seed_count": 2,
            "validation_seed_end": 41001,
        },
        "cpsat_config": {
            "job_horizon": 4,
            "solver_seconds": 0.1,
            "time_scale": 100,
            "random_seed": 0,
            "num_search_workers": 1,
        },
        "validation_metrics": {"solver_fallback_rate_mean": 0.0},
        "selection_data_boundary": {"final_test_used_for_selection": False},
    }


def test_validate_design_enforces_disjoint_selection_ranges() -> None:
    design = validate_design(_design())
    assert design["validation_seed_start"] == 41100
    assert design["final_test_seed_end"] == 42009

    overlapping = _design()
    overlapping["or_tuning_seed_start"] = 41099
    overlapping["or_tuning_seed_count"] = 2
    overlapping["or_tuning_seed_end"] = 41100
    with pytest.raises(ValueError, match="disjoint"):
        validate_design(overlapping)


def test_validate_design_rejects_training_seed_and_final_leakage() -> None:
    leaked_training = _design()
    leaked_training["training_seeds"] = [7, 17, 40000]
    with pytest.raises(ValueError, match="training seeds"):
        validate_design(leaked_training)

    leaked_validation = _design()
    leaked_validation["validation_seed_start"] = 41999
    leaked_validation["validation_seed_count"] = 2
    leaked_validation["validation_seed_end"] = 42000
    with pytest.raises(ValueError, match="validation seeds"):
        validate_design(leaked_validation)


def test_validate_cpsat_freeze_prevents_or_drift() -> None:
    design = validate_design(_design())
    config = validate_cpsat_freeze(_freeze(), design)
    assert config["job_horizon"] == 4
    assert config["solver_seconds"] == 0.1

    drifted = _freeze()
    drifted["selected_policy"] = "FJSP_CPSAT_H8_B100MS"
    with pytest.raises(ValueError, match="does not match"):
        validate_cpsat_freeze(drifted, design)


def _baseline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = {
        "EARLIEST_DUE_DATE": [12.0, 14.0],
        "ROLLING_HORIZON_CPSAT": [9.0, 10.0],
        "SHORTEST_PROCESSING": [18.0, 16.0],
    }
    for policy, wtt in values.items():
        for offset, seed in enumerate((41100, 41101)):
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "validation",
                    "instance_sha256": f"instance-{seed}",
                    "policy": policy,
                    "weighted_tardiness": wtt[offset],
                }
            )
    return rows


def _make_baseline_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    write_csv(_baseline_rows(), root / "baseline_runs.csv")
    write_csv(
        [{"policy": policy} for policy in (
            "EARLIEST_DUE_DATE",
            "ROLLING_HORIZON_CPSAT",
            "SHORTEST_PROCESSING",
        )],
        root / "baseline_summary.csv",
    )


def _make_member_artifact(root: Path, design: dict[str, object], seed: int, wtt: list[float]) -> None:
    member = root / f"seed_{seed}"
    member.mkdir(parents=True)
    (member / "fjsp_maskable_ppo.zip").write_bytes(f"model-{seed}".encode())
    manifest = {
        "algorithm": "MaskablePPO",
        "training_seed": seed,
        "training_config": {**design["training_config"], "seed": seed, "verbose": 0},
        "environment_config": design["environment_config"],
        "training_seconds": float(seed),
        "runtime": {"sb3_contrib": "2.9.0"},
    }
    (member / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    rows = [
        {
            "training_seed": seed,
            "seed": validation_seed,
            "seed_regime": "validation",
            "instance_sha256": f"instance-{validation_seed}",
            "policy": "MASKABLE_PPO",
            "weighted_tardiness": value,
        }
        for validation_seed, value in zip((41100, 41101), wtt, strict=True)
    ]
    write_csv(rows, member / "ppo_validation_runs.csv")
    mean_wtt = float(np.mean(wtt))
    write_csv(
        [
            {
                "training_seed": seed,
                "policy": "MASKABLE_PPO",
                "weighted_tardiness_mean": mean_wtt,
                "weighted_tardiness_std": float(np.std(wtt, ddof=1)),
                "weighted_tardiness_ci_low": min(wtt),
                "weighted_tardiness_ci_high": max(wtt),
                "makespan_mean": 20.0 + seed,
                "mean_flow_time": 10.0 + seed,
                "mean_decision_time_ms": 0.2 + seed / 1000.0,
            }
        ],
        member / "ppo_validation_summary.csv",
    )


def test_artifact_contract_matches_common_instance_panel(tmp_path: Path) -> None:
    design = validate_design(_design())
    baseline_root = tmp_path / "baseline"
    members_root = tmp_path / "members"
    _make_baseline_artifact(baseline_root)
    baseline_rows, _, fingerprints = validate_baseline_panel(design, baseline_root)
    assert fingerprints == {41100: "instance-41100", 41101: "instance-41101"}

    _make_member_artifact(members_root, design, 7, [10.0, 11.0])
    member, rows = validate_member(design, members_root, 7, fingerprints)
    assert member["n_validation_seeds"] == 2
    comparisons = compare_member_to_baselines(
        rows,
        baseline_rows,
        training_seed=7,
        n_bootstrap=100,
        n_permutations=100,
    )
    assert {row["baseline"] for row in comparisons} == {
        "EARLIEST_DUE_DATE",
        "ROLLING_HORIZON_CPSAT",
        "SHORTEST_PROCESSING",
    }


def test_member_artifact_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    design = validate_design(_design())
    members_root = tmp_path / "members"
    _make_member_artifact(members_root, design, 7, [10.0, 11.0])
    with pytest.raises(ValueError, match="different common-seed instances"):
        validate_member(
            design,
            members_root,
            7,
            {41100: "wrong", 41101: "instance-41101"},
        )


def test_representative_member_is_median_role_not_best() -> None:
    members = [
        {"training_seed": 7, "weighted_tardiness_mean": 5.0, "mean_decision_time_ms": 0.3},
        {"training_seed": 17, "weighted_tardiness_mean": 9.0, "mean_decision_time_ms": 0.1},
        {"training_seed": 27, "weighted_tardiness_mean": 7.0, "mean_decision_time_ms": 0.2},
    ]
    selected = select_representative_member(members)
    assert selected["training_seed"] == 27


def test_aggregate_comparison_uses_training_seed_as_inference_unit() -> None:
    rows = []
    for baseline in (
        "EARLIEST_DUE_DATE",
        "ROLLING_HORIZON_CPSAT",
        "SHORTEST_PROCESSING",
    ):
        for training_seed, improvement in ((7, 1.0), (17, -1.0), (27, 2.0)):
            rows.append(
                {
                    "training_seed": training_seed,
                    "baseline": baseline,
                    "mean_improvement": improvement,
                }
            )
    aggregate = aggregate_comparisons(rows, n_bootstrap=100)
    assert len(aggregate) == 3
    assert all(row["n_training_seeds"] == 3 for row in aggregate)
    assert all(row["inference_unit"] == "training_seed" for row in aggregate)


class _FirstFeasibleModel:
    def predict(self, observation, *, action_masks, deterministic):
        del observation, deterministic
        return np.asarray(np.flatnonzero(action_masks)[0]), None


def test_member_runner_uses_validation_seeds_and_masks() -> None:
    generator = FJSPGeneratorConfig(
        n_jobs=3,
        n_machines=2,
        n_families=2,
        operations_min=1,
        operations_max=2,
        eligible_machines_min=1,
        eligible_machines_max=2,
        mean_interarrival=1.0,
        processing_min=1.0,
        processing_max=3.0,
    )
    env_config = FJSPEnvConfig(generator=generator, default_setup_time=0.5)
    rows = evaluate_member(
        _FirstFeasibleModel(),
        training_seed=7,
        seed_start=41100,
        seed_count=2,
        generator_config=generator,
        env_config=env_config,
    )
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {41100, 41101}
    assert {row["training_seed"] for row in rows} == {7}
    assert {row["policy"] for row in rows} == {"MASKABLE_PPO"}


def test_member_runner_rejects_final_seed_access() -> None:
    generator = FJSPGeneratorConfig(
        n_jobs=2,
        n_machines=2,
        operations_min=1,
        operations_max=1,
        eligible_machines_min=1,
        eligible_machines_max=1,
    )
    env_config = FJSPEnvConfig(generator=generator)
    with pytest.raises(ValueError, match="below final seed 42000"):
        evaluate_member(
            _FirstFeasibleModel(),
            training_seed=7,
            seed_start=41999,
            seed_count=2,
            generator_config=generator,
            env_config=env_config,
        )
