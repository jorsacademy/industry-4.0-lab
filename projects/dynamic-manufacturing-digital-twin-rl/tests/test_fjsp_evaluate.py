from __future__ import annotations

import numpy as np
import pytest

from dmdtrl.fjsp_env import FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_evaluate import (
    FJSPEvaluationConfig,
    compare_candidate,
    evaluate_panel,
    instance_fingerprint,
    phase5_seed_regime,
    summarize_panel,
)
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_optimization import FJSPCPSATConfig


class FirstFeasibleModel:
    def predict(self, observation, *, action_masks, deterministic):
        assert deterministic is True
        assert np.asarray(observation).ndim == 1
        valid = np.flatnonzero(action_masks)
        assert valid.size > 0
        return np.asarray(valid[0]), None


def _generator() -> FJSPGeneratorConfig:
    return FJSPGeneratorConfig(
        n_jobs=3,
        n_machines=3,
        n_families=2,
        operations_min=2,
        operations_max=2,
        eligible_machines_min=1,
        eligible_machines_max=2,
        mean_interarrival=1.0,
        processing_min=1.0,
        processing_max=3.0,
    )


def test_phase5_seed_regimes() -> None:
    assert phase5_seed_regime(40000) == "development"
    assert phase5_seed_regime(40999) == "development"
    assert phase5_seed_regime(41000) == "validation"
    assert phase5_seed_regime(41999) == "validation"
    assert phase5_seed_regime(42000) == "final"
    with pytest.raises(ValueError, match="40000"):
        phase5_seed_regime(39999)


def test_env_and_direct_generator_share_exact_seeded_instance() -> None:
    generator = _generator()
    seed = 40021
    direct = generate_fjsp_instance(np.random.default_rng(seed), generator)
    env = FlexibleJobShopEnv(FJSPEnvConfig(generator=generator, default_setup_time=0.5))
    env.reset(seed=seed)
    assert env.simulator is not None
    assert instance_fingerprint(env.simulator.instance) == instance_fingerprint(direct)


def test_common_seed_panel_contains_identical_instance_fingerprint() -> None:
    rows = evaluate_panel(
        FJSPEvaluationConfig(
            seeds=1,
            seed_start=40022,
            default_setup_time=0.5,
            bootstrap=100,
            permutations=100,
        ),
        generator_config=_generator(),
        cpsat_config=FJSPCPSATConfig(job_horizon=3, solver_seconds=0.25),
        maskable_model=FirstFeasibleModel(),
    )
    assert len(rows) == 4
    assert {row["policy"] for row in rows} == {
        "SHORTEST_PROCESSING",
        "EARLIEST_DUE_DATE",
        "ROLLING_HORIZON_CPSAT",
        "MASKABLE_PPO",
    }
    assert len({row["instance_sha256"] for row in rows}) == 1
    assert {row["seed_regime"] for row in rows} == {"development"}
    ppo_row = next(row for row in rows if row["policy"] == "MASKABLE_PPO")
    assert 0.0 < float(ppo_row["unique_action_fraction"]) <= 1.0
    assert float(ppo_row["mean_feasible_actions"]) >= 1.0


def test_summary_and_paired_candidate_comparison() -> None:
    rows = []
    for seed, candidate, baseline in ((40000, 8.0, 10.0), (40001, 9.0, 12.0)):
        common = {
            "seed": seed,
            "seed_regime": "development",
            "instance_sha256": str(seed),
            "makespan": 20.0,
            "mean_flow_time": 10.0,
            "total_setup_time": 1.0,
            "utilization": 0.5,
            "mean_decision_time_ms": 1.0,
            "fallback_rate": 0.0,
            "solver_success_rate": 1.0,
            "unique_action_fraction": 1.0,
            "mean_feasible_actions": 2.0,
        }
        rows.append({**common, "policy": "MASKABLE_PPO", "weighted_tardiness": candidate})
        rows.append({**common, "policy": "EARLIEST_DUE_DATE", "weighted_tardiness": baseline})

    summary = summarize_panel(rows, bootstrap=100)
    assert [row["policy"] for row in summary] == ["MASKABLE_PPO", "EARLIEST_DUE_DATE"]
    comparisons = compare_candidate(
        rows,
        candidate="MASKABLE_PPO",
        bootstrap=100,
        permutations=100,
    )
    assert len(comparisons) == 1
    assert float(comparisons[0]["mean_improvement"]) == pytest.approx(2.5)
    assert float(comparisons[0]["percent_improvement"]) > 0.0


def test_paired_comparison_rejects_mismatched_seed_sets() -> None:
    rows = [
        {"seed": 40000, "policy": "MASKABLE_PPO", "weighted_tardiness": 1.0},
        {"seed": 40001, "policy": "MASKABLE_PPO", "weighted_tardiness": 2.0},
        {"seed": 40000, "policy": "EARLIEST_DUE_DATE", "weighted_tardiness": 3.0},
    ]
    with pytest.raises(ValueError, match="seed sets"):
        compare_candidate(
            rows,
            candidate="MASKABLE_PPO",
            bootstrap=10,
            permutations=10,
        )


def test_evaluation_config_rejects_pre_phase5_seed() -> None:
    with pytest.raises(ValueError, match="40000"):
        FJSPEvaluationConfig(seed_start=39999).validate()
