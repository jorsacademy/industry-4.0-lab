"""Gymnasium environment for stochastic predictive-maintenance decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class MaintenanceEconomics:
    production_margin: float = 12.0
    operating_cost: float = 1.0
    inspection_cost: float = 2.0
    preventive_maintenance_cost: float = 12.0
    replacement_cost: float = 35.0
    failure_cost: float = 80.0
    preventive_downtime_cost: float = 6.0
    replacement_downtime_cost: float = 12.0


class PredictiveMaintenanceEnv(gym.Env):
    """Single-asset condition-based maintenance MDP.

    Observation: [degradation, normalized_age, load, sensor, time_since_maintenance]
    Actions: 0 continue, 1 inspect, 2 maintain, 3 replace.
    """

    metadata = {"render_modes": ["ansi"]}

    CONTINUE = 0
    INSPECT = 1
    MAINTAIN = 2
    REPLACE = 3

    def __init__(
        self,
        horizon: int = 200,
        failure_threshold: float = 1.0,
        seed: int | None = None,
        economics: MaintenanceEconomics | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.failure_threshold = float(failure_threshold)
        self.economics = economics or MaintenanceEconomics()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self._initial_seed = seed
        self._reset_internal_state()

    def _reset_internal_state(self) -> None:
        self.step_count = 0
        self.age = 0
        self.time_since_maintenance = 0
        self.degradation = 0.05
        self.load = 0.5
        self.sensor = self.degradation
        self.failures = 0
        self.inspections = 0
        self.maintenance_actions = 0
        self.replacements = 0
        self.total_cost = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        effective_seed = self._initial_seed if seed is None else seed
        super().reset(seed=effective_seed)
        self._reset_internal_state()
        self.load = float(self.np_random.uniform(0.35, 0.85))
        self.sensor = self._measure_sensor(noise_std=0.04)
        return self._observation(), self._info()

    def _measure_sensor(self, noise_std: float) -> float:
        return float(np.clip(self.degradation + self.np_random.normal(0.0, noise_std), 0.0, 1.0))

    def _observation(self) -> np.ndarray:
        normalized_age = min(self.age / max(self.horizon, 1), 1.0)
        normalized_since = min(self.time_since_maintenance / max(self.horizon, 1), 1.0)
        return np.asarray(
            [self.degradation, normalized_age, self.load, self.sensor, normalized_since],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, float | int]:
        return {
            "degradation": float(self.degradation),
            "failures": self.failures,
            "inspections": self.inspections,
            "maintenance_actions": self.maintenance_actions,
            "replacements": self.replacements,
            "total_cost": float(self.total_cost),
        }

    def _advance_degradation(self) -> None:
        base_wear = 0.006 + 0.010 * self.load
        stochastic_wear = max(0.0, float(self.np_random.normal(0.004, 0.004)))
        self.degradation = float(np.clip(self.degradation + base_wear + stochastic_wear, 0.0, 1.2))

    def _failure_probability(self) -> float:
        if self.degradation >= self.failure_threshold:
            return 1.0
        # Risk accelerates nonlinearly as degradation approaches the threshold.
        risk = 0.001 + 0.12 * (self.degradation / self.failure_threshold) ** 5
        return float(np.clip(risk, 0.0, 1.0))

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..3")

        econ = self.economics
        reward = 0.0
        period_cost = 0.0

        if action in (self.CONTINUE, self.INSPECT):
            reward += econ.production_margin - econ.operating_cost
            period_cost += econ.operating_cost
            self._advance_degradation()

            if action == self.INSPECT:
                self.inspections += 1
                period_cost += econ.inspection_cost
                reward -= econ.inspection_cost
                self.sensor = self._measure_sensor(noise_std=0.01)
            else:
                self.sensor = self._measure_sensor(noise_std=0.05)

            if self.np_random.random() < self._failure_probability():
                self.failures += 1
                period_cost += econ.failure_cost
                reward -= econ.failure_cost
                self.degradation = 0.05
                self.age = 0
                self.time_since_maintenance = 0
                self.sensor = self._measure_sensor(noise_std=0.03)
        elif action == self.MAINTAIN:
            self.maintenance_actions += 1
            period_cost += econ.preventive_maintenance_cost + econ.preventive_downtime_cost
            reward -= econ.preventive_maintenance_cost + econ.preventive_downtime_cost
            self.degradation = max(0.03, self.degradation * 0.30)
            self.time_since_maintenance = 0
            self.sensor = self._measure_sensor(noise_std=0.02)
        else:  # REPLACE
            self.replacements += 1
            period_cost += econ.replacement_cost + econ.replacement_downtime_cost
            reward -= econ.replacement_cost + econ.replacement_downtime_cost
            self.degradation = 0.02
            self.age = 0
            self.time_since_maintenance = 0
            self.sensor = self._measure_sensor(noise_std=0.02)

        self.total_cost += period_cost
        self.step_count += 1
        self.age += 1
        self.time_since_maintenance += 1
        self.load = float(np.clip(0.75 * self.load + 0.25 * self.np_random.uniform(0.25, 1.0), 0.0, 1.0))

        terminated = False
        truncated = self.step_count >= self.horizon
        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> str:
        return (
            f"step={self.step_count} degradation={self.degradation:.3f} "
            f"load={self.load:.2f} failures={self.failures} "
            f"maintenance={self.maintenance_actions} replacements={self.replacements}"
        )
