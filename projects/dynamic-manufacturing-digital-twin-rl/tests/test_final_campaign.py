from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dmdtrl.final_campaign import (
    AGGREGATE_METRICS,
    PPO_AGGREGATE_POLICY,
    average_ppo_by_environment_seed,
    hierarchical_ppo_improvement_ci,
    holm_adjust,
    ppo_policy_name,
    ppo_training_seed_dispersion,
    validate_design,
    verify_frozen_models,
)


def ppo_freeze() -> dict:
    return {
        "status": "validation_complete_representative_frozen",
        "members": [
            {"training_seed": 101, "model_sha256": "", "training_manifest_sha256": ""},
            {"training_seed": 202, "model_sha256": "", "training_manifest_sha256": ""},
            {"training_seed": 303, "model_sha256": "", "training_manifest_sha256": ""},
        ],
    }


def cpsat_freeze() -> dict:
    return {
        "status": "frozen_for_final_evaluation",
        "cpsat_horizon": 8,
        "solver_seconds": 0.1,
    }


def design() -> dict:
    return {
        "status": "predeclared_final_test_design",
        "primary_metric": "weighted_tardiness",
        "primary_fixed_baseline": "WEIGHTED_COMPOSITE",
        "ppo_training_seeds": [101, 202, 303],
        "cpsat": {"horizon": 8, "solver_seconds": 0.1},
        "nominal": {"seed_start": 20000, "seed_count": 100},
        "stress": {
            "seed_start": 30000,
            "seed_count": 100,
            "scenarios": ["demand_120", "compound_stress"],
        },
        "bootstrap": 500,
        "permutations": 1000,
    }


def test_validate_design_enforces_frozen_seed_and_controller_boundaries() -> None:
    validated = validate_design(design(), ppo_freeze(), cpsat_freeze())
    assert validated["nominal"]["seed_start"] == 20000
    assert validated["stress"]["seed_start"] == 30000

    bad = design()
    bad["ppo_training_seeds"] = [101, 202]
    with pytest.raises(ValueError, match="exactly match"):
        validate_design(bad, ppo_freeze(), cpsat_freeze())

    bad = design()
    bad["nominal"] = {"seed_start": 29950, "seed_count": 100}
    with pytest.raises(ValueError, match="below 30000"):
        validate_design(bad, ppo_freeze(), cpsat_freeze())

    bad = design()
    bad["cpsat"] = {"horizon": 12, "solver_seconds": 0.1}
    with pytest.raises(ValueError, match="horizon"):
        validate_design(bad, ppo_freeze(), cpsat_freeze())


def test_verify_frozen_models_checks_hashes_and_manifest_seed(tmp_path: Path) -> None:
    freeze = ppo_freeze()
    for member in freeze["members"]:
        seed = member["training_seed"]
        member_dir = tmp_path / f"seed_{seed}"
        member_dir.mkdir()
        model = member_dir / "ppo_dispatcher.zip"
        manifest = member_dir / "training_manifest.json"
        model.write_bytes(f"model-{seed}".encode())
        manifest.write_text(json.dumps({"training_seed": seed}), encoding="utf-8")
        member["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        member["training_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()

    verified = verify_frozen_models(tmp_path, freeze)
    assert set(verified) == {101, 202, 303}

    (tmp_path / "seed_202" / "ppo_dispatcher.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_models(tmp_path, freeze)


def ppo_rows() -> list[dict]:
    rows: list[dict] = []
    for environment_seed in (20000, 20001):
        for training_seed, base in ((101, 10.0), (202, 20.0), (303, 30.0)):
            row = {
                "scenario": "nominal",
                "policy": ppo_policy_name(training_seed),
                "seed": environment_seed,
            }
            for metric in AGGREGATE_METRICS:
                row[metric] = base + (environment_seed - 20000)
            rows.append(row)
    return rows


def test_training_seed_average_keeps_environment_seed_as_analysis_unit() -> None:
    rows = ppo_rows()
    averaged = average_ppo_by_environment_seed(
        rows,
        [101, 202, 303],
        scenario="nominal",
    )
    assert len(averaged) == 2
    assert averaged[0]["policy"] == PPO_AGGREGATE_POLICY
    assert averaged[0]["training_seed_count"] == 3
    assert averaged[0]["weighted_tardiness"] == pytest.approx(20.0)
    assert averaged[1]["weighted_tardiness"] == pytest.approx(21.0)

    incomplete = rows[:-1]
    with pytest.raises(ValueError, match="incomplete PPO"):
        average_ppo_by_environment_seed(incomplete, [101, 202, 303], scenario="nominal")


def test_training_seed_dispersion_is_reported_separately() -> None:
    result = ppo_training_seed_dispersion(
        ppo_rows(),
        [101, 202, 303],
        scenario="nominal",
    )
    assert result["n_training_seeds"] == 3
    assert result["ppo_wtt_training_seed_mean"] == pytest.approx(20.5)
    assert result["ppo_wtt_training_seed_std"] > 0.0


def test_hierarchical_bootstrap_resamples_training_and_environment_seeds() -> None:
    rows: list[dict] = []
    for environment_seed in range(20):
        rows.append(
            {
                "policy": "BASE",
                "seed": environment_seed,
                "weighted_tardiness": 100.0,
            }
        )
        for training_seed, value in ((101, 80.0), (202, 90.0), (303, 100.0)):
            rows.append(
                {
                    "policy": ppo_policy_name(training_seed),
                    "seed": environment_seed,
                    "weighted_tardiness": value,
                }
            )

    result = hierarchical_ppo_improvement_ci(
        rows,
        [101, 202, 303],
        baseline="BASE",
        n_bootstrap=500,
        seed=7,
    )
    assert result["hierarchical_mean_improvement"] == pytest.approx(10.0)
    assert result["hierarchical_training_seeds"] == 3
    assert result["hierarchical_environment_seeds"] == 20
    assert result["hierarchical_ci_low"] <= 10.0 <= result["hierarchical_ci_high"]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
