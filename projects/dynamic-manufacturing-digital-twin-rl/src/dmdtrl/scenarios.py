from __future__ import annotations

from dataclasses import dataclass, replace

from dmdtrl.env import EnvConfig


@dataclass(frozen=True, slots=True)
class Scenario:
    """A controlled distribution shift applied to the nominal environment."""

    name: str
    description: str
    arrival_intensity_factor: float = 1.0
    breakdown_probability_factor: float = 1.0
    due_date_factor_multiplier: float = 1.0
    machine_speed_multiplier: float = 1.0
    setup_time_multiplier: float = 1.0

    def apply(self, base: EnvConfig | None = None) -> EnvConfig:
        cfg = base or EnvConfig()
        factors = (
            self.arrival_intensity_factor,
            self.breakdown_probability_factor,
            self.due_date_factor_multiplier,
            self.machine_speed_multiplier,
            self.setup_time_multiplier,
        )
        if any(value <= 0.0 for value in factors):
            raise ValueError("scenario multipliers must be positive")

        return replace(
            cfg,
            mean_interarrival=cfg.mean_interarrival / self.arrival_intensity_factor,
            breakdown_probability=min(
                0.95,
                cfg.breakdown_probability * self.breakdown_probability_factor,
            ),
            due_date_factor_min=cfg.due_date_factor_min * self.due_date_factor_multiplier,
            due_date_factor_max=cfg.due_date_factor_max * self.due_date_factor_multiplier,
            machine_speed_min=cfg.machine_speed_min * self.machine_speed_multiplier,
            machine_speed_max=cfg.machine_speed_max * self.machine_speed_multiplier,
            sequence_setup_time=cfg.sequence_setup_time * self.setup_time_multiplier,
        )


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="nominal",
        description="Nominal training-like operating conditions.",
    ),
    Scenario(
        name="demand_120",
        description="20% higher arrival intensity.",
        arrival_intensity_factor=1.20,
    ),
    Scenario(
        name="demand_140",
        description="40% higher arrival intensity.",
        arrival_intensity_factor=1.40,
    ),
    Scenario(
        name="demand_160",
        description="60% higher arrival intensity.",
        arrival_intensity_factor=1.60,
    ),
    Scenario(
        name="breakdown_2x",
        description="Machine breakdown probability doubled.",
        breakdown_probability_factor=2.0,
    ),
    Scenario(
        name="breakdown_4x",
        description="Machine breakdown probability quadrupled.",
        breakdown_probability_factor=4.0,
    ),
    Scenario(
        name="tight_due_085",
        description="Due-date allowances compressed by 15%.",
        due_date_factor_multiplier=0.85,
    ),
    Scenario(
        name="slow_machines_090",
        description="All machine-speed ranges shifted 10% lower.",
        machine_speed_multiplier=0.90,
    ),
    Scenario(
        name="setup_2x",
        description="Sequence-dependent family setup time doubled.",
        setup_time_multiplier=2.0,
    ),
    Scenario(
        name="compound_stress",
        description=(
            "Combined 40% demand increase, 3x breakdown risk, tighter due dates, "
            "slower machines, and 50% higher setup time."
        ),
        arrival_intensity_factor=1.40,
        breakdown_probability_factor=3.0,
        due_date_factor_multiplier=0.85,
        machine_speed_multiplier=0.95,
        setup_time_multiplier=1.50,
    ),
)

SCENARIO_REGISTRY: dict[str, Scenario] = {scenario.name: scenario for scenario in DEFAULT_SCENARIOS}


def select_scenarios(names: list[str] | None = None) -> list[Scenario]:
    if not names:
        return list(DEFAULT_SCENARIOS)
    unknown = sorted(set(names) - SCENARIO_REGISTRY.keys())
    if unknown:
        available = ", ".join(SCENARIO_REGISTRY)
        raise ValueError(f"unknown scenario(s): {', '.join(unknown)}; available: {available}")
    return [SCENARIO_REGISTRY[name] for name in names]
