from dmdtrl.research import fixed_policies


def test_fixed_policies_cover_eight_unique_actions():
    policies = fixed_policies()
    assert len(policies) == 8
    assert len({policy.name for policy in policies}) == 8
    assert {policy.act(None) for policy in policies} == set(range(8))
