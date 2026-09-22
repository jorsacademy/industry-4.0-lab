"""Train DQN or PPO agents on the predictive-maintenance environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .environment import PredictiveMaintenanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise SystemExit('Install RL dependencies with: pip install -e ".[rl]"') from exc

    env = PredictiveMaintenanceEnv(seed=args.seed)
    model_cls = DQN if args.algorithm == "dqn" else PPO
    model = model_cls("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.algorithm}_predictive_maintenance"
    model.save(output)
    print(f"Saved model to {output}.zip")


if __name__ == "__main__":
    main()
