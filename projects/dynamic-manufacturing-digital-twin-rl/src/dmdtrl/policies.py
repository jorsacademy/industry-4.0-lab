from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class DispatchPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def act(self, observation: np.ndarray) -> int: ...


@dataclass(frozen=True, slots=True)
class FixedActionPolicy:
    action: int
    policy_name: str

    @property
    def name(self) -> str:
        return self.policy_name

    def act(self, observation: np.ndarray) -> int:
        del observation
        return int(self.action)


@dataclass(slots=True)
class PredictPolicyAdapter:
    model: Any
    policy_name: str = "PPO"
    deterministic: bool = True

    @property
    def name(self) -> str:
        return self.policy_name

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(np.asarray(action).item())


def load_ppo_policy(model_path: str | Path, *, deterministic: bool = True) -> PredictPolicyAdapter:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('Install RL dependencies with: pip install -e ".[rl]"') from exc
    model = PPO.load(str(model_path))
    return PredictPolicyAdapter(model=model, policy_name="PPO", deterministic=deterministic)
