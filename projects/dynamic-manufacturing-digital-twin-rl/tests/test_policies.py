import numpy as np

from dmdtrl.policies import FixedActionPolicy, PredictPolicyAdapter


class FakeModel:
    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert observation.shape == (2,)
        return np.array(3), None


def test_fixed_action_policy():
    policy = FixedActionPolicy(action=5, policy_name="MIN_SETUP")
    assert policy.name == "MIN_SETUP"
    assert policy.act(np.zeros(2)) == 5


def test_predict_policy_adapter_converts_action_to_int():
    policy = PredictPolicyAdapter(FakeModel())
    assert policy.name == "PPO"
    assert policy.act(np.zeros(2)) == 3
