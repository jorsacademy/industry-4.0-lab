from __future__ import annotations

import pytest

from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_or_selection import select_operating_point, validate_validation_grid
from dmdtrl.fjsp_or_sensitivity import (
    compare_to_reference,
    evaluate_sensitivity,
    sensitivity_grid,
    variant_name,
)


def _small_generator() -> FJSPGeneratorConfig:
    return FJSPGeneratorConfig(
        n_jobs=3,
        n_machines=3,
        n_families=2,
        operations_min=2,
        operations_max=2,
        eligible_machines_min=1,
        eligible_machines_max=2,
        processing_min=1.0,
        processing_max=3.0,
    )


def test_sensitivity_grid_contract() -> None:
    assert variant_name(4, 0.05) == "FJSP_CPSAT_H4_B50MS"
    assert sensitivity_grid((4, 8), (0.05,)) == [(4, 0.05), (8, 0.05)]
    with pytest.raises(ValueError, match="at least two"):
        sensitivity_grid((4,), (0.05,))
    with pytest.raises(ValueError, match="positive"):
        sensitivity_grid((0, 4), (0.05,))


def test_small_real_sensitivity_grid_and_reference_comparison() -> None:
    raw, summary = evaluate_sensitivity(
        seeds=(41000, 41001),
        horizons=(2, 3),
        solver_budgets=(0.10,),
        generator_config=_small_generator(),
        setup_time=0.5,
        bootstrap=50,
    )
    assert len(raw) == 4
    assert len(summary) == 2
    assert {row["seed_regime"] for row in raw} == {"validation"}
    by_seed = {}
    for row in raw:
        by_seed.setdefault(int(row["seed"]), set()).add(str(row["instance_sha256"]))
    assert all(len(fingerprints) == 1 for fingerprints in by_seed.values())
    assert any(bool(row["pareto_optimal"]) for row in summary)

    comparisons = compare_to_reference(
        raw,
        reference_policy=variant_name(3, 0.10),
        bootstrap=50,
        permutations=50,
    )
    assert {row["metric"] for row in comparisons} == {
        "weighted_tardiness",
        "mean_decision_time_ms",
    }
    assert all(int(row["n_pairs"]) == 2 for row in comparisons)


def test_validation_grid_requires_complete_identical_seed_panel() -> None:
    raw = []
    for policy in ("A", "B"):
        for seed in (41000, 41001):
            raw.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "seed_regime": "validation",
                    "instance_sha256": f"instance-{seed}",
                }
            )
    assert validate_validation_grid(raw, seed_start=41000, seed_count=2) == [41000, 41001]

    broken = [dict(row) for row in raw]
    broken[-1]["instance_sha256"] = "different"
    with pytest.raises(ValueError, match="fingerprints"):
        validate_validation_grid(broken, seed_start=41000, seed_count=2)

    with pytest.raises(ValueError, match="41000-41999"):
        validate_validation_grid(raw, seed_start=40999, seed_count=2)


def test_selector_applies_reliability_quality_and_latency_rule() -> None:
    rows = [
        {
            "policy": "fast-bad",
            "weighted_tardiness_mean": 110.0,
            "mean_decision_time_ms_mean": 5.0,
            "solver_fallback_rate_mean": 0.0,
            "cpsat_horizon": 4,
            "solver_budget_ms": 20.0,
            "pareto_optimal": True,
        },
        {
            "policy": "best-quality",
            "weighted_tardiness_mean": 100.0,
            "mean_decision_time_ms_mean": 20.0,
            "solver_fallback_rate_mean": 0.0,
            "cpsat_horizon": 8,
            "solver_budget_ms": 100.0,
            "pareto_optimal": True,
        },
        {
            "policy": "near-best-faster",
            "weighted_tardiness_mean": 101.5,
            "mean_decision_time_ms_mean": 10.0,
            "solver_fallback_rate_mean": 0.005,
            "cpsat_horizon": 8,
            "solver_budget_ms": 50.0,
            "pareto_optimal": True,
        },
        {
            "policy": "unreliable",
            "weighted_tardiness_mean": 99.0,
            "mean_decision_time_ms_mean": 7.0,
            "solver_fallback_rate_mean": 0.05,
            "cpsat_horizon": 12,
            "solver_budget_ms": 20.0,
            "pareto_optimal": True,
        },
    ]
    selected = select_operating_point(
        rows,
        quality_tolerance_pct=2.0,
        max_fallback_rate_pct=1.0,
    )
    assert selected["selected_policy"] == "near-best-faster"
    assert selected["cpsat_horizon"] == 8
    assert selected["solver_budget_ms"] == pytest.approx(50.0)


def test_selector_rejects_no_reliable_pareto_point() -> None:
    rows = [
        {
            "policy": "bad",
            "weighted_tardiness_mean": 10.0,
            "mean_decision_time_ms_mean": 1.0,
            "solver_fallback_rate_mean": 0.2,
            "cpsat_horizon": 4,
            "solver_budget_ms": 20.0,
            "pareto_optimal": True,
        }
    ]
    with pytest.raises(ValueError, match="reliable"):
        select_operating_point(rows, max_fallback_rate_pct=1.0)
