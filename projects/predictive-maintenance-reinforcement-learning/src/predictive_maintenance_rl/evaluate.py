"""Evaluate baseline or trained RL maintenance policies."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .baselines import age_based_policy, condition_based_policy, run_to_failure_policy
from .environment import PredictiveMaintenanceEnv


def evaluate_policy(policy: Callable[[np.ndarray], int], episodes: int = 100, seed: int = 42) -> dict[str, float]:
    rewards: list[float] = []
    costs: list[float] = []
    failures: list[int] = []
    maintenance: list[int] = []
    replacements: list[int] = []
    inspections: list[int] = []
    lengths: list[int] = []

    for episode in range(episodes):
        env = PredictiveMaintenanceEnv(seed=seed + episode)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        info = {}
        while not done:
            action = int(policy(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        costs.append(float(info["total_cost"]))
        failures.append(int(info["failures"]))
        maintenance.append(int(info["maintenance_actions"]))
        replacements.append(int(info["replacements"]))
        inspections.append(int(info["inspections"]))
        lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_cost": float(np.mean(costs)),
        "failure_rate": float(np.mean(np.asarray(failures) > 0)),
        "mean_failures": float(np.mean(failures)),
        "mean_maintenance": float(np.mean(maintenance)),
        "mean_replacements": float(np.mean(replacements)),
        "mean_inspections": float(np.mean(inspections)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def _print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key:>22}: {value:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("baselines", "dqn", "ppo"), default="baselines")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.policy == "baselines":
        policies = {
            "run_to_failure": run_to_failure_policy,
            "age_based": age_based_policy,
            "condition_based": condition_based_policy,
        }
        for name, policy in policies.items():
            _print_metrics(name, evaluate_policy(policy, args.episodes, args.seed))
        return

    if args.model_path is None:
        raise SystemExit("--model-path is required for DQN/PPO evaluation")

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise SystemExit('Install RL dependencies with: pip install -e ".[rl]"') from exc

    model_cls = DQN if args.policy == "dqn" else PPO
    model = model_cls.load(args.model_path)

    def model_policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    _print_metrics(args.policy, evaluate_policy(model_policy, args.episodes, args.seed))


if __name__ == "__main__":
    main()
