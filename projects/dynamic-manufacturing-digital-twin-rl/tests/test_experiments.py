import csv

import numpy as np
import pytest

from dmdtrl.experiments import (
    compare_candidate_to_baselines,
    evaluate_policies,
    paired_comparison,
    summarize_runs,
    write_csv,
)
from dmdtrl.policies import FixedActionPolicy


class FakeEnv:
    def __init__(self):
        self.seed = 0
        self.steps = 0
        self.total_action = 0

    def reset(self, seed=None):
        self.seed = int(seed or 0)
        self.steps = 0
        self.total_action = 0
        return np.array([0.0, 1.0], dtype=np.float32), {}

    def step(self, action):
        self.total_action += int(action)
        self.steps += 1
        terminated = self.steps == 3
        return np.array([0.0, 1.0], dtype=np.float32), 0.0, terminated, False, {}

    def metrics(self):
        return {
            "completed_jobs": 3.0,
            "makespan": 10.0 + self.total_action + self.seed,
            "mean_waiting_time": 2.0 + self.total_action,
            "total_tardiness": 4.0 + self.total_action,
            "weighted_tardiness": 8.0 + self.total_action,
            "total_setup_time": 1.0 + self.total_action,
            "total_repair_time": 0.0,
            "on_time_rate": 1.0 - 0.01 * self.total_action,
            "utilization": 0.8,
            "mean_quality_risk": 0.2,
        }


def factory(config):
    del config
    return FakeEnv()


def test_evaluate_and_summarize_policies():
    policies = [FixedActionPolicy(0, "A"), FixedActionPolicy(1, "B")]
    rows = evaluate_policies(policies, [0, 1, 2], env_factory=factory)
    assert len(rows) == 6
    summaries = summarize_runs(rows, metrics=["weighted_tardiness"], n_bootstrap=500)
    assert {row["policy"] for row in summaries} == {"A", "B"}
    a_summary = next(row for row in summaries if row["policy"] == "A")
    assert a_summary["n_seeds"] == 3
    assert a_summary["weighted_tardiness_mean"] == pytest.approx(8.0)


def test_paired_comparison_positive_means_candidate_is_better():
    policies = [FixedActionPolicy(0, "candidate"), FixedActionPolicy(1, "baseline")]
    rows = evaluate_policies(policies, range(6), env_factory=factory)
    result = paired_comparison(
        rows,
        candidate="candidate",
        baseline="baseline",
        metric="weighted_tardiness",
        n_bootstrap=500,
        n_permutations=1_000,
    )
    assert result["mean_improvement"] == pytest.approx(3.0)
    assert result["percent_improvement"] > 0.0
    assert result["probability_of_superiority"] == pytest.approx(1.0)


def test_compare_candidate_to_baselines_handles_max_metric():
    policies = [FixedActionPolicy(0, "candidate"), FixedActionPolicy(1, "baseline")]
    rows = evaluate_policies(policies, range(4), env_factory=factory)
    results = compare_candidate_to_baselines(
        rows,
        candidate="candidate",
        baselines=["baseline"],
        metrics=["on_time_rate"],
        n_bootstrap=300,
        n_permutations=500,
    )
    assert len(results) == 1
    assert results[0]["direction"] == "max"
    assert results[0]["mean_improvement"] > 0.0


def test_write_csv_supports_heterogeneous_controller_fields(tmp_path):
    output = tmp_path / "mixed.csv"
    rows = [
        {"policy": "fixed", "seed": 1, "weighted_tardiness": 10.0},
        {
            "policy": "CP_SAT_RH",
            "seed": 1,
            "weighted_tardiness": 9.0,
            "solver_fallback_rate": 0.0,
        },
    ]

    write_csv(rows, output)

    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert "solver_fallback_rate" in written[0]
    assert written[0]["solver_fallback_rate"] == ""
    assert written[1]["solver_fallback_rate"] == "0.0"
