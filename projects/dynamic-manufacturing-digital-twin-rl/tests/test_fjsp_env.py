from __future__ import annotations

import numpy as np
import pytest

from dmdtrl.fjsp_env import FJSPActionCodec, FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_models import FJSPAction


def _config() -> FJSPEnvConfig:
    return FJSPEnvConfig(
        generator=FJSPGeneratorConfig(
            n_jobs=5,
            n_machines=3,
            n_families=3,
            operations_min=2,
            operations_max=3,
            eligible_machines_min=1,
            eligible_machines_max=2,
        )
    )


def test_action_codec_round_trip() -> None:
    codec = FJSPActionCodec(job_ids=(10, 20), max_operations=3, n_machines=4)
    action = FJSPAction(job_id=20, operation_index=2, machine_id=3)
    assert codec.decode(codec.encode(action)) == action
    with pytest.raises(ValueError, match="outside"):
        codec.decode(codec.size)


def test_reset_observation_and_mask_are_consistent() -> None:
    env = FlexibleJobShopEnv(_config())
    observation, info = env.reset(seed=123)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,)
    assert mask.dtype == bool
    assert mask.sum() == info["feasible_action_count"]
    assert mask.sum() > 0

    simulator_actions = set(env.simulator.eligible_actions())
    mask_actions = {env.codec.decode(idx) for idx in np.flatnonzero(mask)}
    assert mask_actions == simulator_actions


def test_mask_preserves_precedence_and_trace_records_decision() -> None:
    env = FlexibleJobShopEnv(_config())
    env.reset(seed=7)
    mask = env.action_masks()
    action_id = int(np.flatnonzero(mask)[0])
    action = env.codec.decode(action_id)
    assert action.operation_index == 0

    observation, reward, terminated, truncated, info = env.step(action_id)
    assert observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert truncated is False
    assert terminated is False
    assert info["decision_count"] == 1
    trace = env.decision_trace_records()[0]
    assert trace["action_id"] == action_id
    assert trace["job_id"] == action.job_id
    assert trace["operation_index"] == 0
    assert trace["feasible_action_count"] == int(mask.sum())

    next_index = env.simulator.next_operation[action.job_id]
    assert next_index == 1
    if env.simulator.operation_ready_at[action.job_id] > env.simulator.current_time + 1e-12:
        assert not any(
            candidate.job_id == action.job_id and candidate.operation_index == 1
            for candidate in env.simulator.eligible_actions()
        )


def test_infeasible_action_is_rejected() -> None:
    env = FlexibleJobShopEnv(_config())
    env.reset(seed=99)
    mask = env.action_masks()
    invalid = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError, match="infeasible"):
        env.step(invalid)
