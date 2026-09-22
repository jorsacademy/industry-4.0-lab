from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dmdtrl.fjsp_env import FJSPEnvConfig, FlexibleJobShopEnv
from dmdtrl.fjsp_generator import FJSPGeneratorConfig


@dataclass(slots=True, frozen=True)
class FJSPMaskablePPOConfig:
    total_timesteps: int = 100_000
    seed: int = 601
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
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
            raise ValueError("batch_size must be in [1, n_steps]")
        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.ent_coef < 0.0:
            raise ValueError("ent_coef must be non-negative")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")

    def algorithm_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "seed": self.seed,
            "device": self.device,
            "verbose": self.verbose,
            "policy_kwargs": {
                "net_arch": {
                    "pi": [self.hidden_units, self.hidden_units],
                    "vf": [self.hidden_units, self.hidden_units],
                }
            },
        }


def model_archive_path(output: str | Path) -> Path:
    target = Path(output)
    if target.suffix == ".zip":
        return target
    return Path(f"{target}.zip")


def _save_base_path(output: str | Path) -> Path:
    return model_archive_path(output).with_suffix("")


def default_manifest_path(output: str | Path) -> Path:
    archive = model_archive_path(output)
    return archive.with_name(f"{archive.stem}_manifest.json")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def validate_mask_contract(env: FlexibleJobShopEnv, *, seed: int) -> dict[str, int]:
    env.reset(seed=seed)
    decisions = 0
    min_feasible = env.action_space.n
    max_feasible = 0
    while True:
        mask = env.action_masks()
        if mask.shape != (env.action_space.n,):
            raise RuntimeError("action mask shape does not match action space")
        feasible = int(mask.sum())
        if feasible <= 0:
            if env.simulator is not None and env.simulator.terminated:
                break
            raise RuntimeError("non-terminal FJSP state has no feasible masked action")
        min_feasible = min(min_feasible, feasible)
        max_feasible = max(max_feasible, feasible)
        action_id = int(np.flatnonzero(mask)[0])
        _, _, terminated, truncated, _ = env.step(action_id)
        decisions += 1
        if truncated:
            raise RuntimeError("FJSP environment unexpectedly truncated during mask validation")
        if terminated:
            break
    return {
        "validated_decisions": decisions,
        "min_feasible_actions": min_feasible,
        "max_feasible_actions": max_feasible,
    }


def build_training_manifest(
    *,
    training_config: FJSPMaskablePPOConfig,
    env_config: FJSPEnvConfig,
    model_path: Path,
    training_seconds: float,
    mask_contract: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "training_seed": training_config.seed,
        "training_config": asdict(training_config),
        "environment_config": asdict(env_config),
        "model_path": str(model_path),
        "training_seconds": float(training_seconds),
        "mask_contract": dict(mask_contract),
        "phase5_seed_policy": {
            "training_randomness": "separate from evaluation seeds",
            "development_evaluation": "40000-40999",
            "validation": "41000-41999",
            "final": "42000+ (range frozen before access)",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "project": package_version("dynamic-manufacturing-digital-twin-rl"),
            "sb3_contrib": package_version("sb3-contrib"),
            "stable_baselines3": package_version("stable-baselines3"),
            "torch": package_version("torch"),
            "gymnasium": package_version("gymnasium"),
            "numpy": package_version("numpy"),
        },
        "git": {
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_maskable_ppo(
    output: str | Path,
    *,
    training_config: FJSPMaskablePPOConfig | None = None,
    env_config: FJSPEnvConfig | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError("sb3-contrib is required for Phase-5 Maskable PPO training") from exc

    training = training_config or FJSPMaskablePPOConfig()
    training.validate()
    environment = env_config or FJSPEnvConfig()
    environment.validate()

    contract_env = FlexibleJobShopEnv(environment)
    mask_contract = validate_mask_contract(contract_env, seed=training.seed)
    contract_env.close()

    env = FlexibleJobShopEnv(environment)
    model = MaskablePPO("MlpPolicy", env, **training.algorithm_kwargs())
    started = perf_counter()
    model.learn(total_timesteps=training.total_timesteps)
    training_seconds = perf_counter() - started

    archive = model_archive_path(output)
    archive.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(_save_base_path(output)))
    env.close()

    manifest_path = Path(metadata_path) if metadata_path else default_manifest_path(output)
    manifest = build_training_manifest(
        training_config=training,
        env_config=environment,
        model_path=archive,
        training_seconds=training_seconds,
        mask_contract=mask_contract,
    )
    write_manifest(manifest_path, manifest)
    return archive, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase-5 Maskable PPO on the FJSP environment.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--operations-min", type=int, default=2)
    parser.add_argument("--operations-max", type=int, default=4)
    parser.add_argument("--eligible-max", type=int, default=3)
    parser.add_argument("--setup-time", type=float, default=1.0)
    parser.add_argument("--output", default="models/fjsp_maskable_ppo")
    parser.add_argument("--metadata")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training = FJSPMaskablePPOConfig(
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
    environment = FJSPEnvConfig(
        generator=FJSPGeneratorConfig(
            n_jobs=args.jobs,
            n_machines=args.machines,
            n_families=4,
            operations_min=args.operations_min,
            operations_max=args.operations_max,
            eligible_machines_min=1,
            eligible_machines_max=min(args.eligible_max, args.machines),
        ),
        default_setup_time=args.setup_time,
    )
    archive, manifest = train_maskable_ppo(
        args.output,
        training_config=training,
        env_config=environment,
        metadata_path=args.metadata,
    )
    print(f"Maskable PPO model saved to {archive}")
    print(f"Training manifest saved to {manifest}")


if __name__ == "__main__":
    main()
