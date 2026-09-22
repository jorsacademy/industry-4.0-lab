from __future__ import annotations

import csv
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import dmdtrl.fjsp_hh_evaluate as hh_evaluate
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_hh_evaluate import (
    FJSPHyperHeuristicDevelopmentConfig,
    evaluate_development_panel,
)
from dmdtrl.fjsp_operators import FJSPOperator
from dmdtrl.fjsp_optimization import FJSPCPSATConfig


class FixedOperatorModel:
    def __init__(self, operator: FJSPOperator) -> None:
        self.operator = operator

    def predict(self, observation, deterministic: bool = True):  # noqa: ANN001, ANN201
        assert deterministic
        assert observation is not None
        return np.asarray(int(self.operator)), None


def _generator() -> FJSPGeneratorConfig:
    return FJSPGeneratorConfig(
        n_jobs=5,
        n_machines=3,
        n_families=3,
        operations_min=2,
        operations_max=3,
        eligible_machines_min=1,
        eligible_machines_max=2,
    )


def test_development_seed_boundary_rejects_validation_access() -> None:
    FJSPHyperHeuristicDevelopmentConfig(seeds=2, seed_start=40000).validate()
    with pytest.raises(ValueError, match="41000"):
        FJSPHyperHeuristicDevelopmentConfig(seeds=2, seed_start=40999).validate()
    with pytest.raises(ValueError, match="40000"):
        FJSPHyperHeuristicDevelopmentConfig(seeds=1, seed_start=39999).validate()
    with pytest.raises(ValueError, match="seeds"):
        FJSPHyperHeuristicDevelopmentConfig(seeds=0).validate()
    with pytest.raises(ValueError, match="non-negative"):
        FJSPHyperHeuristicDevelopmentConfig(default_setup_time=-0.1).validate()
    with pytest.raises(ValueError, match="bootstrap"):
        FJSPHyperHeuristicDevelopmentConfig(bootstrap=0).validate()
    with pytest.raises(ValueError, match="bootstrap"):
        FJSPHyperHeuristicDevelopmentConfig(permutations=0).validate()


def test_development_panel_uses_common_instances_for_all_controllers() -> None:
    evaluation = FJSPHyperHeuristicDevelopmentConfig(
        seeds=2,
        seed_start=40040,
        default_setup_time=0.5,
        bootstrap=100,
        permutations=100,
    )
    rows = evaluate_development_panel(
        evaluation,
        generator_config=_generator(),
        cpsat_config=FJSPCPSATConfig(job_horizon=4, solver_seconds=0.05),
        ppo_model=FixedOperatorModel(FJSPOperator.EARLIEST_DUE_DATE),
    )

    expected_policies = {operator.name for operator in FJSPOperator}
    expected_policies.update({"ROLLING_HORIZON_CPSAT", "PPO_HYPER_HEURISTIC"})
    assert len(rows) == 2 * len(expected_policies)
    assert {str(row["policy"]) for row in rows} == expected_policies
    assert {int(row["seed"]) for row in rows} == {40040, 40041}
    assert {str(row["seed_regime"]) for row in rows} == {"development"}

    by_seed: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    for seed_rows in by_seed.values():
        assert len(seed_rows) == len(expected_policies)
        assert len({str(row["instance_sha256"]) for row in seed_rows}) == 1

    ppo_rows = [row for row in rows if row["policy"] == "PPO_HYPER_HEURISTIC"]
    assert all(float(row["unique_operator_fraction"]) > 0.0 for row in ppo_rows)
    assert all(float(row["mean_underlying_feasible_actions"]) >= 1.0 for row in ppo_rows)


def test_load_ppo_uses_stable_baselines_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("stable_baselines3")
    sentinel = object()

    class FakePPO:
        @classmethod
        def load(cls, path: str, device: str = "cpu"):  # noqa: ANN206
            assert path == "model.zip"
            assert device == "cpu"
            return sentinel

    fake_module.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_module)
    assert hh_evaluate.load_ppo("model.zip", device="cpu") is sentinel


def test_cli_main_writes_declared_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.csv"
    comparisons_path = tmp_path / "comparisons.csv"

    fake_rows = [
        {
            "seed": 40060,
            "seed_regime": "development",
            "instance_sha256": "abc",
            "policy": "ROLLING_HORIZON_CPSAT",
            "weighted_tardiness": 10.0,
            "makespan": 20.0,
            "mean_flow_time": 8.0,
            "mean_decision_time_ms": 5.0,
            "fallback_rate": 0.0,
        },
        {
            "seed": 40060,
            "seed_regime": "development",
            "instance_sha256": "abc",
            "policy": "EARLIEST_DUE_DATE",
            "weighted_tardiness": 15.0,
            "makespan": 21.0,
            "mean_flow_time": 9.0,
            "mean_decision_time_ms": 0.01,
            "fallback_rate": 0.0,
        },
    ]
    fake_summary = [
        {
            "policy": "ROLLING_HORIZON_CPSAT",
            "weighted_tardiness_mean": 10.0,
            "mean_decision_time_ms": 5.0,
        },
        {
            "policy": "EARLIEST_DUE_DATE",
            "weighted_tardiness_mean": 15.0,
            "mean_decision_time_ms": 0.01,
        },
    ]
    fake_comparisons = [
        {
            "candidate": "ROLLING_HORIZON_CPSAT",
            "baseline": "EARLIEST_DUE_DATE",
            "metric": "weighted_tardiness",
            "mean_improvement": 5.0,
        }
    ]

    monkeypatch.setattr(
        hh_evaluate,
        "evaluate_development_panel",
        lambda *args, **kwargs: fake_rows,
    )
    monkeypatch.setattr(
        hh_evaluate,
        "summarize_panel",
        lambda rows, bootstrap: fake_summary,
    )
    monkeypatch.setattr(
        hh_evaluate,
        "compare_candidate",
        lambda rows, **kwargs: fake_comparisons,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fjsp-hh-evaluate",
            "--seeds",
            "1",
            "--seed-start",
            "40060",
            "--jobs",
            "5",
            "--machines",
            "3",
            "--operations-min",
            "2",
            "--operations-max",
            "3",
            "--eligible-max",
            "2",
            "--bootstrap",
            "10",
            "--permutations",
            "10",
            "--raw-output",
            str(raw_path),
            "--summary-output",
            str(summary_path),
            "--comparisons-output",
            str(comparisons_path),
        ],
    )

    hh_evaluate.main()

    assert raw_path.exists()
    assert summary_path.exists()
    assert comparisons_path.exists()
    with raw_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    output = capsys.readouterr().out
    assert "ROLLING_HORIZON_CPSAT" in output
    assert "EARLIEST_DUE_DATE" in output
