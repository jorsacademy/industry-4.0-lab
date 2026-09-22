import numpy as np

from predictive_maintenance_rl.environment import PredictiveMaintenanceEnv


def test_reset_observation_is_valid():
    env = PredictiveMaintenanceEnv(seed=7, horizon=10)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["failures"] == 0
    assert info["total_cost"] == 0.0


def test_seeded_resets_are_reproducible():
    env_a = PredictiveMaintenanceEnv(seed=123)
    env_b = PredictiveMaintenanceEnv(seed=123)
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    np.testing.assert_allclose(obs_a, obs_b)


def test_episode_truncates_at_horizon():
    env = PredictiveMaintenanceEnv(seed=1, horizon=5)
    env.reset()
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(env.MAINTAIN)
        assert not terminated
    assert truncated


def test_preventive_maintenance_reduces_degradation():
    env = PredictiveMaintenanceEnv(seed=2, horizon=20)
    env.reset()
    env.degradation = 0.8
    _, reward, _, _, info = env.step(env.MAINTAIN)
    assert env.degradation < 0.3
    assert reward < 0
    assert info["maintenance_actions"] == 1


def test_replacement_resets_asset_condition():
    env = PredictiveMaintenanceEnv(seed=3, horizon=20)
    env.reset()
    env.degradation = 0.95
    obs, reward, _, _, info = env.step(env.REPLACE)
    assert obs[0] < 0.1
    assert reward < 0
    assert info["replacements"] == 1
