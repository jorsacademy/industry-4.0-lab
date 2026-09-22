from __future__ import annotations

import pytest

from dmdtrl.env import EnvConfig
from dmdtrl.or_sensitivity import (
    _mark_pareto,
    compare_to_reference,
    evaluate_sensitivity,
    sensitivity_grid,
    variant_name,
)


def test_variant_name_is_stable_and_validates_inputs() -> None:
    assert variant_name(12, 0.10) == "CP_SAT_H12_B100000US"
    with pytest.raises(ValueError):
        variant_name(0, 0.10)
    with pytest.raises(ValueError):
        variant_name(12, 0.0)


def test_sensitivity_grid_rejects_invalid_or_duplicate_points() -> None:
    assert sensitivity_grid([4, 8], [0.05]) == [(4, 0.05), (8, 0.05)]
    with pytest.raises(ValueError):
        sensitivity_grid([], [0.05])
    with pytest.raises(ValueError):
        sensitivity_grid([4], [-0.01])
    with pytest.raises(ValueError, match="duplicate"):
        sensitivity_grid([4, 4], [0.05])
    with pytest.raises(ValueError, match="at least two"):
        sensitivity_grid([4], [0.05])


def test_pareto_marking_uses_quality_and_latency() -> None:
    rows = [
        {
            "policy": "fast",
            "weighted_tardiness_mean": 12.0,
            "mean_decision_time_ms_mean": 2.0,
        },
        {
            "policy": "quality",
            "weighted_tardiness_mean": 8.0,
            "mean_decision_time_ms_mean": 8.0,
        },
        {
            "policy": "dominated",
            "weighted_tardiness_mean": 13.0,
            "mean_decision_time_ms_mean": 9.0,
        },
    ]

    marked = _mark_pareto(rows)
    flags = {row["policy"]: row["pareto_optimal"] for row in marked}
    assert flags == {"fast": True, "quality": True, "dominated": False}


def test_compare_to_reference_is_seed_paired_for_quality_and_latency() -> None:
    rows = []
    for seed in range(4):
        rows.extend(
            [
                {
                    "policy": "candidate",
                    "seed": seed,
                    "weighted_tardiness": 8.0,
                    "mean_decision_time_ms": 4.0,
                },
                {
                    "policy": "reference",
                    "seed": seed,
                    "weighted_tardiness": 10.0,
                    "mean_decision_time_ms": 5.0,
                },
            ]
        )

    comparisons = compare_to_reference(
        rows,
        reference_policy="reference",
        n_bootstrap=100,
        n_permutations=200,
    )
    assert {row["metric"] for row in comparisons} == {
        "weighted_tardiness",
        "mean_decision_time_ms",
    }
    assert all(float(row["mean_improvement"]) > 0.0 for row in comparisons)


def test_small_actual_sensitivity_run_retains_configuration_fields() -> None:
    pytest.importorskip("ortools")
    raw, summary = evaluate_sensitivity(
        [77],
        [3, 4],
        [0.05],
        EnvConfig(
            n_jobs=6,
            n_machines=2,
            mean_interarrival=0.2,
            breakdown_probability=0.0,
        ),
        n_bootstrap=50,
    )

    assert len(raw) == 2
    assert len(summary) == 2
    assert {int(row["cpsat_horizon"]) for row in raw} == {3, 4}
    assert all("pareto_optimal" in row for row in summary)
