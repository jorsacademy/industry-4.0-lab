from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dmdtrl.fjsp_env import FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_operators import OPERATOR_NAMES, FJSPOperator, select_operator_action
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


class FlexibleJobShopHyperHeuristicEnv(gym.Env[np.ndarray, int]):
    """PPO-ready FJSP environment where actions select dispatch operators.

    The underlying simulator transition, reward function, observation encoder,
    and instance generator are delegated to ``FlexibleJobShopEnv``. The learned
    controller therefore changes only the action abstraction: one small,
    always-feasible operator ID is translated into one concrete FJSP assignment.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: FJSPEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.base_env = FlexibleJobShopEnv(config=config, render_mode=render_mode)
        self.action_space = spaces.Discrete(len(FJSPOperator))
        self.observation_space = self.base_env.observation_space
        self.render_mode = render_mode
        self.operator_trace: list[dict[str, float | int | bool | str]] = []

    @property
    def config(self) -> FJSPEnvConfig:
        return self.base_env.config

    @property
    def simulator(self) -> FlexibleJobShopSimulator | None:
        return self.base_env.simulator

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.base_env.reset(seed=seed, options=options)
        self.operator_trace = []
        return observation, self._decorate_info(info)

    def action_masks(self) -> np.ndarray:
        """All operators are feasible whenever a simulator decision exists."""

        mask = np.ones(self.action_space.n, dtype=bool)
        if self.simulator is not None and self.simulator.terminated:
            mask[:] = False
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        operator_id = int(action)
        if not self.action_space.contains(operator_id):
            raise ValueError(
                f"operator action {operator_id} is outside [0, {self.action_space.n})"
            )
        simulator = self.simulator
        if simulator is None:
            raise RuntimeError("environment is not reset")
        if simulator.terminated:
            raise RuntimeError("schedule is already complete")

        operator = FJSPOperator(operator_id)
        eligible_count = len(simulator.eligible_actions())
        selected = select_operator_action(simulator, operator)
        flat_action_id = self.base_env.codec.encode(selected)
        observation, reward, terminated, truncated, info = self.base_env.step(
            flat_action_id
        )
        base_trace = self.base_env.decision_trace_records()[-1]
        record: dict[str, float | int | bool | str] = dict(base_trace)
        record.update(
            {
                "operator_action_id": operator_id,
                "operator_name": operator.name,
                "selected_flat_action_id": flat_action_id,
                "underlying_feasible_action_count": eligible_count,
            }
        )
        self.operator_trace.append(record)

        decorated = self._decorate_info(info)
        decorated.update(
            {
                "operator_action_id": operator_id,
                "operator_name": operator.name,
                "selected_flat_action_id": flat_action_id,
            }
        )
        return observation, reward, terminated, truncated, decorated

    def _decorate_info(self, info: dict[str, Any]) -> dict[str, Any]:
        decorated = dict(info)
        decorated["controller"] = "FJSP_HYPER_HEURISTIC"
        decorated["operator_action_count"] = self.action_space.n
        decorated["operator_names"] = OPERATOR_NAMES
        return decorated

    def decision_trace_records(self) -> list[dict[str, float | int | bool | str]]:
        return [dict(record) for record in self.operator_trace]

    def render(self) -> str | None:
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()
