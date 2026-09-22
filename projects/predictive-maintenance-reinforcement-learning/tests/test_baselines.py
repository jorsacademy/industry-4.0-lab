import numpy as np

from predictive_maintenance_rl.baselines import age_based_policy, condition_based_policy, run_to_failure_policy


def obs(degradation=0.2, age=0.1, sensor=0.2):
    return np.asarray([degradation, age, 0.5, sensor, 0.1], dtype=np.float32)


def test_run_to_failure_always_continues():
    assert run_to_failure_policy(obs(sensor=0.99)) == 0


def test_age_policy_maintains_after_threshold():
    assert age_based_policy(obs(age=0.2), age_threshold=0.45) == 0
    assert age_based_policy(obs(age=0.5), age_threshold=0.45) == 2


def test_condition_policy_uses_maintenance_and_replacement_thresholds():
    assert condition_based_policy(obs(sensor=0.4)) == 0
    assert condition_based_policy(obs(sensor=0.7)) == 2
    assert condition_based_policy(obs(sensor=0.95)) == 3
