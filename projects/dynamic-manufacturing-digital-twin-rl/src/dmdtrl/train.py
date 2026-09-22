from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from dmdtrl.env import DynamicManufacturingEnv, EnvConfig


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    total_timesteps: int = 150_000
    seed: int = 42
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    hidden_units: int = 128
    device: str = "cpu"
    verbose: int = 1

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_steps <= 1:
            raise ValueError("n_steps must be greater than one")
        if self.batch_size <= 0 or self.batch_size > self.n_steps:
            raise ValueError("batch_size must be positive and no larger than n_steps")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")
        if self.ent_coef < 0.0:
            raise ValueError("ent_coef must be non-negative")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")

    def ppo_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "policy_kwargs": {"net_arch": [self.hidden_units, self.hidden_units]},
            "device": self.device,
            "verbose": self.verbose,
            "seed": self.seed,
        }


def model_archive_path(output: Path) -> Path:
    return output if output.suffix == ".zip" else Path(f"{output}.zip")


def default_manifest_path(output: Path) -> Path:
    archive = model_archive_path(output)
    return archive.with_name(f"{archive.stem}_manifest.json")


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def build_training_manifest(
    training: PPOTrainingConfig,
    env_config: EnvConfig,
    *,
    model_path: Path,
    training_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "training_regime": "nominal_environment",
        "training_seed": training.seed,
        "training_config": asdict(training),
        "environment_config": asdict(env_config),
        "model_path": str(model_path),
        "training_seconds": float(training_seconds),
        "git": {
            "sha": os.getenv("GITHUB_SHA") or os.getenv("GIT_COMMIT"),
            "ref": os.getenv("GITHUB_REF"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "dynamic-manufacturing-digital-twin-rl": _package_version(
                    "dynamic-manufacturing-digital-twin-rl"
                ),
                "stable-baselines3": _package_version("stable-baselines3"),
                "torch": _package_version("torch"),
                "gymnasium": _package_version("gymnasium"),
                "numpy": _package_version("numpy"),
            },
        },
        "evaluation_seed_convention": {
            "validation_seed_start": 10_000,
            "nominal_test_seed_start": 20_000,
            "stress_test_seed_start": 30_000,
            "note": (
                "Validation/model-selection seeds and final-test seeds must remain disjoint from "
                "training randomness and from one another."
            ),
        },
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_ppo(
    training: PPOTrainingConfig,
    *,
    env_config: EnvConfig | None = None,
    output: Path = Path("models/ppo_dispatcher"),
    manifest_output: Path | None = None,
) -> tuple[Path, Path]:
    """Train PPO on the nominal simulator and persist model + audit manifest."""
    training.validate()
    cfg = env_config or EnvConfig()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('Install RL dependencies with: pip install -e ".[rl]"') from exc

    check_env(DynamicManufacturingEnv(cfg), warn=True)
    env = DynamicManufacturingEnv(cfg)
    env.reset(seed=training.seed)
    model = PPO("MlpPolicy", env, **training.ppo_kwargs())

    started = perf_counter()
    model.learn(total_timesteps=training.total_timesteps, progress_bar=False)
    training_seconds = perf_counter() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    archive = model_archive_path(output)
    manifest_path = manifest_output or default_manifest_path(output)
    manifest = build_training_manifest(
        training,
        cfg,
        model_path=archive,
        training_seconds=training_seconds,
    )
    write_manifest(manifest, manifest_path)
    env.close()
    return archive, manifest_path


def main() -> None:  # pragma: no cover - RL integration is exercised in GitHub Actions
    parser = argparse.ArgumentParser(description="Train PPO as a dispatching hyper-heuristic.")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/ppo_dispatcher"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    training = PPOTrainingConfig(
        total_timesteps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        hidden_units=args.hidden_units,
        device=args.device,
        verbose=0 if args.quiet else 1,
    )
    try:
        archive, manifest = train_ppo(
            training,
            output=args.output,
            manifest_output=args.metadata,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Saved PPO model to {archive}")
    print(f"Saved training manifest to {manifest}")


if __name__ == "__main__":  # pragma: no cover
    main()
