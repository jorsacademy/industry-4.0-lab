from __future__ import annotations

import numpy as np
import pytest

from dmdtrl.fjsp_env import FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_hyperheuristic_env import FlexibleJobShopHyperHeuristicEnv
from dmdtrl.fjsp_operators import OPERATOR_NAMES, FJSPOperator, select_operator_action


def _config() -> FJSPEnvConfig:
    return FJSPEnvConfig(
        generator=FJSPGeneratorConfig(
            n_jobs=6,
            n_machines=4,
            n_families=3,
            operations_min=2,
            operations_max=4,
            eligible_machines_min=1,
            eligible_machines_max=3,
        ),
        default_setup_time=1.0,
    )


def test_operator_ids_are_stable_and_complete() -> None:
    assert [int(operator) for operator in FJSPOperator] == list(range(8))
    assert OPERATOR_NAMES == (
        "EARLIEST_DUE_DATE",
        "SHORTEST_PROCESSING",
        "MINIMUM_SETUP",
        "HIGHEST_PRIORITY",
        "MINIMUM_SLACK",
        "CRITICAL_RATIO",
        "SAME_FAMILY_FIRST",
        "WEIGHTED_TARDINESS_RISK",
    )


def test_every_operator_maps_to_a_feasible_assignment() -> None:
    env = FlexibleJobShopEnv(_config())
    env.reset(seed=40123)
    simulator = env.simulator
    assert simulator is not None
    eligible = set(simulator.eligible_actions())
    assert eligible

    for operator in FJSPOperator:
        selected = select_operator_action(simulator, operator)
        assert selected in eligible


def test_hyperheuristic_env_has_small_always_feasible_action_space() -> None:
    env = FlexibleJobShopHyperHeuristicEnv(_config())
    observation, info = env.reset(seed=40124)

    assert observation.shape == env.observation_space.shape
    assert env.action_space.n == len(FJSPOperator)
    assert info["controller"] == "FJSP_HYPER_HEURISTIC"
    assert info["operator_action_count"] == len(FJSPOperator)
    np.testing.assert_array_equal(
        env.action_masks(),
        np.ones(len(FJSPOperator), dtype=bool),
    )


def test_operator_step_reuses_flat_env_transition_and_reward() -> None:
    config = _config()
    flat_env = FlexibleJobShopEnv(config)
    hyper_env = FlexibleJobShopHyperHeuristicEnv(config)
    flat_observation, _ = flat_env.reset(seed=40125)
    hyper_observation, _ = hyper_env.reset(seed=40125)
    np.testing.assert_allclose(flat_observation, hyper_observation, rtol=0.0, atol=0.0)

    simulator = flat_env.simulator
    assert simulator is not None
    selected = select_operator_action(simulator, FJSPOperator.EARLIEST_DUE_DATE)
    flat_action_id = flat_env.codec.encode(selected)

    flat_next, flat_reward, flat_done, flat_truncated, _ = flat_env.step(flat_action_id)
    hyper_next, hyper_reward, hyper_done, hyper_truncated, info = hyper_env.step(
        int(FJSPOperator.EARLIEST_DUE_DATE)
    )

    np.testing.assert_allclose(flat_next, hyper_next, rtol=0.0, atol=0.0)
    assert hyper_reward == pytest.approx(flat_reward)
    assert hyper_done is flat_done
    assert hyper_truncated is flat_truncated
    assert hyper_env.simulator is not None
    assert flat_env.simulator is not None
    assert hyper_env.simulator.schedule[-1] == flat_env.simulator.schedule[-1]
    assert info["selected_flat_action_id"] == flat_action_id

    trace = hyper_env.decision_trace_records()[-1]
    assert trace["operator_action_id"] == int(FJSPOperator.EARLIEST_DUE_DATE)
    assert trace["operator_name"] == "EARLIEST_DUE_DATE"
    assert trace["selected_flat_action_id"] == flat_action_id
    assert trace["underlying_feasible_action_count"] > 0


def test_operator_cycle_completes_episode_without_infeasible_actions() -> None:
    env = FlexibleJobShopHyperHeuristicEnv(_config())
    env.reset(seed=40126)
    decisions = 0

    while True:
        operator = decisions % len(FJSPOperator)
        mask = env.action_masks()
        assert mask.shape == (len(FJSPOperator),)
        assert mask[operator]
        _, _, terminated, truncated, info = env.step(operator)
        decisions += 1
        assert truncated is False
        assert info["operator_action_count"] == len(FJSPOperator)
        if terminated:
            break

    assert env.simulator is not None
    assert env.simulator.terminated
    assert decisions == env.simulator.instance.total_operations
    assert len(env.decision_trace_records()) == decisions
    assert not env.action_masks().any()


def test_out_of_range_operator_is_rejected() -> None:
    env = FlexibleJobShopHyperHeuristicEnv(_config())
    env.reset(seed=40127)
    with pytest.raises(ValueError, match="outside"):
        env.step(len(FJSPOperator))
