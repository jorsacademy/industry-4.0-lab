from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import dmdtrl.fjsp_final as fjsp_final
from dmdtrl.fjsp_final import (
    CPSAT_POLICY,
    FINAL_SEED_END,
    FINAL_SEED_START,
    build_manifest,
    build_parser,
    final_seeds,
    run_final_benchmark,
    validate_final_design,
    validate_final_rows,
)
from dmdtrl.fjsp_operators import OPERATOR_NAMES


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _frozen_inputs() -> tuple[dict, dict, dict, dict]:
    return (
        _load("configs/fjsp_final_baseline_test.json"),
        _load("configs/fjsp_hh_validation_design.json"),
        _load("configs/fjsp_cpsat_validation_freeze.json"),
        _load("configs/fjsp_hh_validation_decision.json"),
    )


def test_final_design_matches_frozen_phase5_boundaries() -> None:
    config, environment, cpsat, decision = _frozen_inputs()

    validate_final_design(config, environment, cpsat, decision)
    assert final_seeds(config) == list(range(FINAL_SEED_START, FINAL_SEED_END + 1))
    assert config["policies"] == [*OPERATOR_NAMES, CPSAT_POLICY]
    assert config["authorization"]["retuning_after_final_access"] is False
    assert decision["decision"]["promote_to_phase5_final"] is False


def test_final_seed_range_cannot_drift() -> None:
    config = _load("configs/fjsp_final_baseline_test.json")
    config["seed_start"] = 41999
    with pytest.raises(ValueError, match="seed range"):
        final_seeds(config)


def test_rejected_rl_cannot_be_inserted_into_final_panel() -> None:
    config, environment, cpsat, decision = _frozen_inputs()
    config["policies"].append("PPO_HYPER_HEURISTIC")
    with pytest.raises(ValueError, match="policy panel drifted"):
        validate_final_design(config, environment, cpsat, decision)


def test_final_design_rejects_reopened_selection() -> None:
    config, environment, cpsat, decision = _frozen_inputs()
    config["authorization"]["retuning_after_final_access"] = True
    with pytest.raises(ValueError, match="retuning"):
        validate_final_design(config, environment, cpsat, decision)


def test_final_design_rejects_promoted_rl_decision() -> None:
    config, environment, cpsat, decision = _frozen_inputs()
    decision["decision"]["promote_to_phase5_final"] = True
    with pytest.raises(ValueError, match="rejected"):
        validate_final_design(config, environment, cpsat, decision)


def test_synthetic_final_rows_require_complete_common_instance_panel() -> None:
    config = _load("configs/fjsp_final_baseline_test.json")
    rows = []
    for seed in range(FINAL_SEED_START, FINAL_SEED_END + 1):
        fingerprint = f"instance-{seed}"
        for policy in config["policies"]:
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "final",
                    "instance_sha256": fingerprint,
                    "policy": policy,
                }
            )
    validate_final_rows(config, rows)

    rows[-1]["instance_sha256"] = "mismatch"
    with pytest.raises(ValueError, match="canonical instance"):
        validate_final_rows(config, rows)


def test_final_rows_reject_missing_policy_episode() -> None:
    config = _load("configs/fjsp_final_baseline_test.json")
    rows = []
    for seed in range(FINAL_SEED_START, FINAL_SEED_END + 1):
        for policy in config["policies"]:
            rows.append(
                {
                    "seed": seed,
                    "seed_regime": "final",
                    "instance_sha256": f"instance-{seed}",
                    "policy": policy,
                }
            )
    rows.pop()
    with pytest.raises(ValueError, match="duplicate or missing"):
        validate_final_rows(config, rows)


def test_build_manifest_marks_final_block_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/28/merge")
    monkeypatch.setenv("GITHUB_WORKFLOW", "FJSP Final Frozen Baselines")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    config = _load("configs/fjsp_final_baseline_test.json")
    summary = [{"policy": CPSAT_POLICY, "weighted_tardiness_mean": 1.0}]
    comparisons = [{"candidate": CPSAT_POLICY, "baseline": "EARLIEST_DUE_DATE"}]

    manifest = build_manifest(config, summary, comparisons)

    assert manifest["status"] == "final_frozen_baseline_test_complete"
    assert manifest["final_block_consumed"] is True
    assert manifest["no_retuning_after_final"] is True
    assert manifest["git"]["sha"] == "abc123"
    assert manifest["summary"] == summary
    assert manifest["comparisons"] == comparisons


def test_run_final_benchmark_orchestrates_without_opening_final_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    synthetic_rows = [
        {
            "seed": FINAL_SEED_START,
            "seed_regime": "synthetic-unit-test",
            "instance_sha256": "not-a-final-instance",
            "policy": CPSAT_POLICY,
            "weighted_tardiness": 1.0,
        }
    ]
    synthetic_summary = [
        {
            "policy": CPSAT_POLICY,
            "n_seeds": 100,
            "weighted_tardiness_mean": 1.0,
            "mean_decision_time_ms": 1.0,
        }
    ]
    synthetic_comparisons = [
        {
            "candidate": CPSAT_POLICY,
            "baseline": "EARLIEST_DUE_DATE",
            "metric": "weighted_tardiness",
            "mean_improvement": 1.0,
        }
    ]

    monkeypatch.setattr(
        fjsp_final,
        "evaluate_final_panel",
        lambda config, environment_design, cpsat_freeze: synthetic_rows,
    )
    monkeypatch.setattr(
        fjsp_final,
        "summarize_panel",
        lambda rows, bootstrap: synthetic_summary,
    )
    monkeypatch.setattr(
        fjsp_final,
        "compare_candidate",
        lambda rows, **kwargs: synthetic_comparisons,
    )

    rows, summary, comparisons, manifest = run_final_benchmark(
        config_path="configs/fjsp_final_baseline_test.json",
        environment_design_path="configs/fjsp_hh_validation_design.json",
        cpsat_freeze_path="configs/fjsp_cpsat_validation_freeze.json",
        rl_decision_path="configs/fjsp_hh_validation_decision.json",
        output_root=tmp_path,
        bootstrap=10,
        permutations=20,
    )

    assert rows == synthetic_rows
    assert summary == synthetic_summary
    assert comparisons == synthetic_comparisons
    assert manifest["final_block_consumed"] is True
    assert (tmp_path / "fjsp_final_runs.csv").exists()
    assert (tmp_path / "fjsp_final_summary.csv").exists()
    assert (tmp_path / "fjsp_final_cpsat_comparisons.csv").exists()
    assert (tmp_path / "fjsp_final_manifest.json").exists()


def test_parser_defaults_are_frozen() -> None:
    args = build_parser().parse_args([])
    assert args.config == "configs/fjsp_final_baseline_test.json"
    assert args.bootstrap == 5_000
    assert args.permutations == 10_000


def test_main_prints_frozen_summary_without_running_real_final(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = [
        {
            "policy": CPSAT_POLICY,
            "weighted_tardiness_mean": 12.5,
            "mean_decision_time_ms": 20.0,
        }
    ]
    comparisons = [
        {
            "baseline": "EARLIEST_DUE_DATE",
            "mean_improvement": 3.0,
            "ci_low": 1.0,
            "ci_high": 5.0,
        }
    ]
    monkeypatch.setattr(
        fjsp_final,
        "run_final_benchmark",
        lambda **kwargs: ([], summary, comparisons, {}),
    )
    monkeypatch.setattr(sys, "argv", ["fjsp_final"])

    fjsp_final.main()

    output = capsys.readouterr().out
    assert "ROLLING_HORIZON_CPSAT" in output
    assert "vs EARLIEST_DUE_DATE" in output
