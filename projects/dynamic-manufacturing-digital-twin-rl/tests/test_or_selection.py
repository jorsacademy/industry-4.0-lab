from __future__ import annotations

import pytest

from dmdtrl.or_selection import select_operating_point, validate_validation_seeds


def test_validate_validation_seeds_requires_declared_validation_range() -> None:
    rows = [
        {"seed": 10000, "policy": "A"},
        {"seed": 10001, "policy": "A"},
        {"seed": 10000, "policy": "B"},
        {"seed": 10001, "policy": "B"},
    ]
    assert validate_validation_seeds(rows, seed_start=10000, seed_count=2) == [10000, 10001]

    with pytest.raises(ValueError, match="10000"):
        validate_validation_seeds(rows, seed_start=9999, seed_count=2)
    with pytest.raises(ValueError, match="20000"):
        validate_validation_seeds(rows, seed_start=19999, seed_count=2)
    with pytest.raises(ValueError, match="do not match"):
        validate_validation_seeds(rows, seed_start=10000, seed_count=3)


def test_validate_validation_seeds_rejects_partial_configuration() -> None:
    rows = [
        {"seed": 10000, "policy": "A"},
        {"seed": 10001, "policy": "A"},
        {"seed": 10000, "policy": "B"},
    ]
    with pytest.raises(ValueError, match="'B'.*do not match"):
        validate_validation_seeds(rows, seed_start=10000, seed_count=2)


def test_select_operating_point_prefers_latency_within_quality_tolerance() -> None:
    rows = [
        {
            "policy": "best-quality",
            "weighted_tardiness_mean": "100.0",
            "mean_decision_time_ms_mean": "12.0",
            "solver_fallback_rate_mean": "0.0",
            "cpsat_horizon": "12",
            "solver_budget_ms": "100.0",
            "pareto_optimal": "True",
        },
        {
            "policy": "fast-near-best",
            "weighted_tardiness_mean": "101.5",
            "mean_decision_time_ms_mean": "4.0",
            "solver_fallback_rate_mean": "0.002",
            "cpsat_horizon": "8",
            "solver_budget_ms": "50.0",
            "pareto_optimal": "True",
        },
        {
            "policy": "too-far",
            "weighted_tardiness_mean": "103.0",
            "mean_decision_time_ms_mean": "1.0",
            "solver_fallback_rate_mean": "0.0",
            "cpsat_horizon": "4",
            "solver_budget_ms": "20.0",
            "pareto_optimal": "True",
        },
        {
            "policy": "dominated",
            "weighted_tardiness_mean": "100.5",
            "mean_decision_time_ms_mean": "2.0",
            "solver_fallback_rate_mean": "0.0",
            "cpsat_horizon": "6",
            "solver_budget_ms": "20.0",
            "pareto_optimal": "False",
        },
    ]

    selected = select_operating_point(rows, quality_tolerance_pct=2.0)
    assert selected["selected_policy"] == "fast-near-best"
    assert selected["cpsat_horizon"] == 8
    assert selected["solver_seconds"] == pytest.approx(0.05)
    assert selected["acceptable_pareto_configurations"] == 2
    assert selected["solver_fallback_rate_mean"] == pytest.approx(0.002)


def test_select_operating_point_excludes_unreliable_fast_configuration() -> None:
    rows = [
        {
            "policy": "reliable",
            "weighted_tardiness_mean": "100.0",
            "mean_decision_time_ms_mean": "8.0",
            "solver_fallback_rate_mean": "0.005",
            "cpsat_horizon": "8",
            "solver_budget_ms": "50.0",
            "pareto_optimal": "True",
        },
        {
            "policy": "unreliable-fast",
            "weighted_tardiness_mean": "99.0",
            "mean_decision_time_ms_mean": "2.0",
            "solver_fallback_rate_mean": "0.05",
            "cpsat_horizon": "12",
            "solver_budget_ms": "20.0",
            "pareto_optimal": "True",
        },
    ]

    selected = select_operating_point(rows, max_fallback_rate_pct=1.0)
    assert selected["selected_policy"] == "reliable"
    assert selected["reliable_pareto_configurations"] == 1


def test_select_operating_point_rejects_invalid_summary() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_operating_point([])
    with pytest.raises(ValueError, match="non-negative"):
        select_operating_point([{"policy": "A"}], quality_tolerance_pct=-1.0)
    with pytest.raises(ValueError, match="between 0 and 100"):
        select_operating_point([{"policy": "A"}], max_fallback_rate_pct=101.0)
    with pytest.raises(ValueError, match="missing fields"):
        select_operating_point([{"policy": "A"}])


def test_select_operating_point_requires_reliable_pareto_configuration() -> None:
    rows = [
        {
            "policy": "A",
            "weighted_tardiness_mean": "10",
            "mean_decision_time_ms_mean": "1",
            "solver_fallback_rate_mean": "0.02",
            "cpsat_horizon": "4",
            "solver_budget_ms": "20",
            "pareto_optimal": "True",
        }
    ]
    with pytest.raises(ValueError, match="fallback-rate"):
        select_operating_point(rows, max_fallback_rate_pct=1.0)
