from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dmdtrl.final_campaign import (
    _annotate_policy_rows,
    apply_primary_holm,
    average_ppo_by_environment_seed,
    build_manifest,
    hierarchical_ppo_improvement_ci,
    holm_adjust,
    ppo_training_seed_dispersion,
    validate_design,
    verify_frozen_models,
    write_csv,
)


def _freeze() -> dict:
    return {
        "status": "validation_complete_representative_frozen",
        "source_workflow_run_id": 123,
        "representative_model": {"training_seed": 101},
        "members": [
            {"training_seed": 101, "model_sha256": "a", "training_manifest_sha256": "b"},
            {"training_seed": 202, "model_sha256": "c", "training_manifest_sha256": "d"},
            {"training_seed": 303, "model_sha256": "e", "training_manifest_sha256": "f"},
        ],
    }


def _cpsat() -> dict:
    return {
        "status": "frozen_for_final_evaluation",
        "cpsat_horizon": 8,
        "solver_seconds": 0.1,
        "provenance": {"artifact_sha256": "deadbeef"},
    }


def _design() -> dict:
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
        "bootstrap": 100,
        "permutations": 200,
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda d: d.update(status="draft"), "predeclared"),
        (lambda d: d.update(primary_metric="makespan"), "primary metric"),
        (lambda d: d.update(primary_fixed_baseline="FIFO"), "fixed baseline"),
        (lambda d: d.update(nominal={"seed_start": 19999, "seed_count": 1}), "at or above 20000"),
        (lambda d: d.update(stress={"seed_start": 29999, "seed_count": 1, "scenarios": ["demand_120"]}), "at or above 30000"),
        (lambda d: d.update(stress={"seed_start": 30000, "seed_count": 1, "scenarios": []}), "at least one stress"),
        (lambda d: d.update(stress={"seed_start": 30000, "seed_count": 1, "scenarios": ["demand_120", "demand_120"]}), "unique"),
        (lambda d: d.update(stress={"seed_start": 30000, "seed_count": 1, "scenarios": ["nominal"]}), "dedicated nominal"),
        (lambda d: d.update(cpsat={"horizon": 8, "solver_seconds": 0.2}), "solve budget"),
        (lambda d: d.update(bootstrap=0), "bootstrap"),
    ],
)
def test_design_rejects_predeclared_boundary_violations(mutator, message: str) -> None:
    design = _design()
    mutator(design)
    with pytest.raises(ValueError, match=message):
        validate_design(design, _freeze(), _cpsat())


def test_design_requires_frozen_controller_states() -> None:
    ppo = _freeze()
    ppo["status"] = "validation_incomplete"
    with pytest.raises(ValueError, match="PPO validation"):
        validate_design(_design(), ppo, _cpsat())

    cpsat = _cpsat()
    cpsat["status"] = "candidate"
    with pytest.raises(ValueError, match="CP-SAT operating point"):
        validate_design(_design(), _freeze(), cpsat)


def test_verify_frozen_models_rejects_missing_and_manifest_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing frozen PPO"):
        verify_frozen_models(tmp_path, _freeze())

    freeze = {
        "status": "validation_complete_representative_frozen",
        "members": [{"training_seed": 101, "model_sha256": "", "training_manifest_sha256": ""}],
    }
    member = tmp_path / "seed_101"
    member.mkdir()
    model = member / "ppo_dispatcher.zip"
    manifest = member / "training_manifest.json"
    model.write_bytes(b"model")
    manifest.write_text(json.dumps({"training_seed": 999}), encoding="utf-8")
    freeze["members"][0]["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    freeze["members"][0]["training_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="manifest seed mismatch"):
        verify_frozen_models(tmp_path, freeze)

    with pytest.raises(ValueError, match="contains no members"):
        verify_frozen_models(tmp_path, {"members": []})


def test_aggregate_panel_rejects_duplicates_and_absence() -> None:
    with pytest.raises(ValueError, match="no PPO rows"):
        average_ppo_by_environment_seed([], [101, 202, 303], scenario="nominal")

    row = {"policy": "PPO_TRAIN_101", "seed": 1}
    duplicated = [row, dict(row)]
    with pytest.raises(ValueError, match="duplicate PPO row"):
        average_ppo_by_environment_seed(duplicated, [101], scenario="nominal")


def test_training_seed_dispersion_rejects_wrong_panel() -> None:
    with pytest.raises(ValueError, match="does not match frozen"):
        ppo_training_seed_dispersion(
            [{"policy": "PPO_TRAIN_101", "weighted_tardiness": 1.0, "mean_decision_time_ms": 0.1}],
            [101, 202],
            scenario="nominal",
        )


def test_hierarchical_bootstrap_validates_direction_and_baseline() -> None:
    with pytest.raises(ValueError, match="direction"):
        hierarchical_ppo_improvement_ci([], [101], baseline="BASE", direction="sideways")
    with pytest.raises(ValueError, match="baseline"):
        hierarchical_ppo_improvement_ci(
            [{"policy": "PPO_TRAIN_101", "seed": 1, "weighted_tardiness": 1.0}],
            [101],
            baseline="BASE",
            n_bootstrap=10,
        )


def test_holm_application_marks_only_primary_wtt_family() -> None:
    rows = [
        {
            "analysis": "ppo_training_seed_average_vs_primary_baseline",
            "metric": "weighted_tardiness",
            "p_value": 0.01,
        },
        {
            "analysis": "ppo_training_seed_average_vs_primary_baseline",
            "metric": "makespan",
            "p_value": 0.02,
        },
    ]
    apply_primary_holm(rows)
    assert rows[0]["primary_hypothesis"] is True
    assert rows[0]["p_value_holm"] == pytest.approx(0.01)
    assert rows[1]["primary_hypothesis"] is False
    assert rows[1]["p_value_holm"] == ""
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust([1.1])


def test_policy_annotation_preserves_controller_class_and_training_seed() -> None:
    rows = [
        {"policy": "PPO_TRAIN_101", "seed": 1},
        {"policy": "CP_SAT_RH", "seed": 1},
        {"policy": "FIFO", "seed": 1},
    ]
    annotated = _annotate_policy_rows(rows, scenario="nominal", training_seeds=[101])
    assert annotated[0]["controller_class"] == "PPO"
    assert annotated[0]["training_seed"] == 101
    assert annotated[1]["controller_class"] == "CP_SAT"
    assert annotated[2]["controller_class"] == "FIXED_RULE"


def test_write_csv_and_manifest_provenance(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "nested" / "rows.csv"
    write_csv([{"a": 1, "b": 2}, {"a": 3, "c": 4}], output)
    text = output.read_text(encoding="utf-8")
    assert "a,b,c" in text
    with pytest.raises(ValueError, match="empty CSV"):
        write_csv([], tmp_path / "empty.csv")

    monkeypatch.setenv("GITHUB_SHA", "abc123")
    manifest = build_manifest(_design(), _freeze(), _cpsat())
    assert manifest["status"] == "final_test_complete"
    assert manifest["git_sha"] == "abc123"
    assert manifest["no_final_test_tuning"] is True
    assert manifest["ppo_freeze_source"]["source_workflow_run_id"] == 123
    assert manifest["cpsat_freeze_source"]["horizon"] == 8
