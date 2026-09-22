from __future__ import annotations

import json
from pathlib import Path

DECISION_PATH = Path("configs/fjsp_hh_validation_decision.json")


def _decision() -> dict:
    return json.loads(DECISION_PATH.read_text(encoding="utf-8"))


def test_operator_selection_v1_is_not_promoted_to_final() -> None:
    decision = _decision()
    assert decision["status"] == "validation_not_promoted_to_final"
    assert decision["architecture"] == "operator_selection_v1"
    assert decision["decision"]["promote_to_phase5_final"] is False
    assert decision["decision"]["final_test_execution_authorized"] is False
    assert decision["data_boundary"]["final_test_used_for_selection"] is False
    assert decision["data_boundary"]["final_test_seed_start"] == 42_000
    assert decision["data_boundary"]["final_test_seed_end"] == 42_099


def test_validation_evidence_uses_all_five_training_seeds() -> None:
    decision = _decision()
    validation = decision["ppo_validation"]
    assert validation["training_seeds"] == [901, 1901, 2901, 3901, 4901]
    assert validation["n_training_seeds"] == 5
    assert validation["representative_training_seed"] == 1901
    assert validation["representative_model_role"] == "deployment_and_demo_only"


def test_key_strong_baseline_intervals_are_below_zero() -> None:
    decision = _decision()
    comparisons = decision["training_seed_level_comparisons"]
    for name in (
        "vs_WEIGHTED_TARDINESS_RISK",
        "vs_EARLIEST_DUE_DATE",
        "vs_ROLLING_HORIZON_CPSAT",
    ):
        comparison = comparisons[name]
        assert comparison["mean_improvement"] < 0.0
        assert comparison["ci_low"] < 0.0
        assert comparison["ci_high"] < 0.0
        assert comparison["training_seed_win_fraction"] == 0.0


def test_future_iteration_cannot_reuse_consumed_validation_block() -> None:
    decision = _decision()
    boundary = decision["next_iteration_boundary"]
    assert boundary["architecture_name"] == "operator_selection_v2"
    assert boundary["may_use_existing_development_data"] is True
    assert boundary["may_reuse_41200_41229_for_selection"] is False
    assert boundary["requires_new_predeclared_validation_block"] is True
    assert boundary["final_42000_42099_remains_embargoed"] is True
