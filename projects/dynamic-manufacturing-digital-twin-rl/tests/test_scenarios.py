import pytest

from dmdtrl.env import EnvConfig
from dmdtrl.scenarios import SCENARIO_REGISTRY, Scenario, select_scenarios


def test_nominal_scenario_preserves_environment_config():
    base = EnvConfig()
    assert SCENARIO_REGISTRY["nominal"].apply(base) == base


def test_demand_shift_changes_interarrival_rate_by_intensity():
    base = EnvConfig(mean_interarrival=1.5)
    shifted = SCENARIO_REGISTRY["demand_140"].apply(base)
    assert shifted.mean_interarrival == pytest.approx(1.5 / 1.4)


def test_compound_stress_changes_multiple_operating_conditions():
    base = EnvConfig()
    shifted = SCENARIO_REGISTRY["compound_stress"].apply(base)
    assert shifted.mean_interarrival < base.mean_interarrival
    assert shifted.breakdown_probability > base.breakdown_probability
    assert shifted.due_date_factor_min < base.due_date_factor_min
    assert shifted.machine_speed_min < base.machine_speed_min
    assert shifted.sequence_setup_time > base.sequence_setup_time


def test_breakdown_probability_is_capped():
    scenario = Scenario(
        name="extreme_failure",
        description="Test cap.",
        breakdown_probability_factor=100.0,
    )
    assert scenario.apply(EnvConfig(breakdown_probability=0.2)).breakdown_probability == 0.95


def test_scenario_rejects_non_positive_multiplier():
    scenario = Scenario(name="invalid", description="Invalid.", machine_speed_multiplier=0.0)
    with pytest.raises(ValueError):
        scenario.apply()


def test_select_scenarios_preserves_requested_order_and_rejects_unknown_names():
    scenarios = select_scenarios(["setup_2x", "nominal"])
    assert [scenario.name for scenario in scenarios] == ["setup_2x", "nominal"]
    with pytest.raises(ValueError, match="unknown scenario"):
        select_scenarios(["does_not_exist"])
