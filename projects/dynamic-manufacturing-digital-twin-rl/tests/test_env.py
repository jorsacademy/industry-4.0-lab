import numpy as np
import pytest

from dmdtrl.dispatch import DispatchRule
from dmdtrl.env import DynamicManufacturingEnv, EnvConfig


def _small_config() -> EnvConfig:
    return EnvConfig(
        n_jobs=20,
        n_machines=3,
        breakdown_probability=0.10,
    )


def test_observation_and_action_spaces_are_respected():
    env = DynamicManufacturingEnv(_small_config())
    obs, info = env.reset(seed=42)

    assert env.observation_space.contains(obs)
    assert env.action_space.n == 8
    assert info["completed_jobs"] == 0

    next_obs, reward, terminated, truncated, info = env.step(int(DispatchRule.FIFO))
    assert env.observation_space.contains(next_obs)
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["completed_jobs"] == 1


def test_complete_episode_schedules_every_job_exactly_once():
    env = DynamicManufacturingEnv(_small_config())
    env.reset(seed=123)

    terminated = False
    while not terminated:
        _, _, terminated, truncated, _ = env.step(int(DispatchRule.WEIGHTED_COMPOSITE))
        assert not truncated

    records = env.schedule_records()
    assert len(records) == 20
    assert len({r["job_id"] for r in records}) == 20
    assert env.metrics()["completed_jobs"] == 20.0
    assert 0.0 <= env.metrics()["on_time_rate"] <= 1.0
    assert 0.0 <= env.metrics()["utilization"] <= 1.0


def test_reset_with_same_seed_reproduces_job_stream_and_first_transition():
    env_a = DynamicManufacturingEnv(_small_config())
    env_b = DynamicManufacturingEnv(_small_config())

    obs_a, _ = env_a.reset(seed=99)
    obs_b, _ = env_b.reset(seed=99)
    np.testing.assert_allclose(obs_a, obs_b)
    assert env_a.jobs == env_b.jobs

    out_a = env_a.step(int(DispatchRule.EARLIEST_DUE_DATE))
    out_b = env_b.step(int(DispatchRule.EARLIEST_DUE_DATE))
    np.testing.assert_allclose(out_a[0], out_b[0])
    assert out_a[1:] == out_b[1:]


def test_step_after_termination_raises():
    env = DynamicManufacturingEnv(EnvConfig(n_jobs=1, n_machines=1, breakdown_probability=0.0))
    env.reset(seed=1)
    _, _, terminated, _, _ = env.step(0)
    assert terminated
    with pytest.raises(RuntimeError):
        env.step(0)
