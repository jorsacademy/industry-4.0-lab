from __future__ import annotations

import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_project_is_complete_and_final_block_is_closed() -> None:
    completion = _load("configs/project_completion.json")

    assert completion["status"] == "PROJECT_COMPLETE"
    assert completion["completion_scope"]["further_model_selection"] is False
    assert completion["completion_scope"]["further_final_test_access"] is False
    assert completion["phase5_final_benchmark"]["final_block_consumed"] is True
    assert completion["phase5_final_benchmark"]["no_retuning_after_final"] is True
    assert completion["phase5_final_benchmark"]["final_seed_start"] == 42000
    assert completion["phase5_final_benchmark"]["final_seed_end"] == 42099
    assert completion["final_conclusion"]["project_state"] == "portfolio_ready_no_further_research_required"


def test_final_winner_and_artifact_provenance_are_frozen() -> None:
    completion = _load("configs/project_completion.json")
    final = completion["phase5_final_benchmark"]
    ranking = final["ranking"]

    assert final["source_pull_request"] == 28
    assert final["workflow_run_id"] == 33271196539
    assert final["artifact_id"] == 9720170969
    assert final["artifact_sha256"] == "8cdef3fb72df4f88156043b1e2aa54daf3f0ad92493385dca58bf1e814a4abf9"
    assert final["winner"] == "ROLLING_HORIZON_CPSAT"
    assert final["frozen_winner_configuration"] == "FJSP_CPSAT_H4_B100MS"
    assert ranking[0]["policy"] == "ROLLING_HORIZON_CPSAT"
    assert ranking[0]["weighted_tardiness_mean"] == 23.467215669499446
    assert ranking[1]["policy"] == "MINIMUM_SLACK"
    assert ranking[1]["weighted_tardiness_mean"] == 30.413395846297632
    assert [row["rank"] for row in ranking] == list(range(1, 10))
    assert [row["weighted_tardiness_mean"] for row in ranking] == sorted(
        row["weighted_tardiness_mean"] for row in ranking
    )
    assert all(row["fallback_rate"] == 0.0 for row in ranking)


def test_key_final_comparison_is_positive_and_predeclared() -> None:
    completion = _load("configs/project_completion.json")
    final = completion["phase5_final_benchmark"]
    min_slack = final["key_paired_comparisons"]["vs_MINIMUM_SLACK"]
    final_design = _load("configs/fjsp_final_baseline_test.json")

    assert min_slack["mean_wtt_improvement"] == 6.946180176798184
    assert min_slack["percent_improvement"] == 22.83921273343692
    assert min_slack["ci_low"] > 0.0
    assert min_slack["ci_high"] > min_slack["ci_low"]
    assert min_slack["n_pairs"] == 100
    assert final_design["authorization"]["selection_complete_before_final_access"] is True
    assert final_design["authorization"]["retuning_after_final_access"] is False
    assert set(final_design["excluded_from_final"]) == {"MASKABLE_PPO", "PPO_HYPER_HEURISTIC"}


def test_both_rl_formulations_remain_rejected() -> None:
    completion = _load("configs/project_completion.json")
    direct = _load("configs/fjsp_direct_ppo_validation_freeze.json")
    hyper = _load("configs/fjsp_hh_validation_decision.json")
    rl = completion["rl_evidence"]

    assert direct["decision"]["accept_for_final_test"] is False
    assert hyper["decision"]["promote_to_phase5_final"] is False
    assert rl["direct_action_maskable_ppo"]["accepted_for_final"] is False
    assert rl["operator_selection_ppo_v1"]["promoted_to_final"] is False
    assert rl["final_rl_test_performed"] is False
    assert rl["direct_action_maskable_ppo"]["vs_cpsat_ci_high"] < 0.0
    assert rl["operator_selection_ppo_v1"]["vs_cpsat_ci_high"] < 0.0


def test_v2_is_archived_and_not_part_of_completed_evidence() -> None:
    completion = _load("configs/project_completion.json")
    archived = completion["archived_extension"]

    assert archived["name"] == "operator_selection_v2"
    assert archived["status"] == "archived_without_merge_or_validation"
    assert archived["pull_request"] == 27
    assert archived["validation_block_41300_41329_opened"] is False
    assert archived["part_of_completed_project"] is False


def test_completion_docs_mark_project_complete() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    report = Path("docs/final_project_report.md").read_text(encoding="utf-8")
    roadmap = Path("docs/roadmap.md").read_text(encoding="utf-8")

    assert "Status: COMPLETE" in readme
    assert "23.4672" in readme
    assert "22.84%" in readme
    assert "Status: COMPLETE" in report
    assert "Project status: COMPLETE" in roadmap
