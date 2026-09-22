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

from dmdtrl.fjsp_env import FJSPEnvConfig
from dmdtrl.fjsp_generator import FJSPGeneratorConfig
from dmdtrl.fjsp_hyperheuristic_env import FlexibleJobShopHyperHeuristicEnv
from dmdtrl.fjsp_operators import OPERATOR_NAMES


@dataclass(slots=True, frozen=True)
class FJSPHyperHeuristicPPOConfig:
    total_timesteps: int = 50_000
    seed: int = 801
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
    return target if target.suffix == ".zip" else Path(f"{target}.zip")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def validate_operator_contract(
    env: FlexibleJobShopHyperHeuristicEnv,
    *,
    seed: int,
) -> dict[str, int]:
    env.reset(seed=seed)
    decisions = 0
    operator_count = int(env.action_space.n)
    while True:
        mask = env.action_masks()
        if mask.shape != (operator_count,):
            raise RuntimeError("operator mask shape does not match action space")
        if not bool(mask.all()):
            if env.simulator is not None and env.simulator.terminated:
                break
            raise RuntimeError("all hyper-heuristic operators must be feasible")
        action = decisions % operator_count
        _, _, terminated, truncated, _ = env.step(action)
        decisions += 1
        if truncated:
            raise RuntimeError("hyper-heuristic environment unexpectedly truncated")
        if terminated:
            break
    return {
        "validated_decisions": decisions,
        "operator_action_count": operator_count,
    }


def build_training_manifest(
    *,
    training_config: FJSPHyperHeuristicPPOConfig,
    env_config: FJSPEnvConfig,
    model_path: Path,
    training_seconds: float,
    operator_contract: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "PPO",
        "controller": "FJSP_HYPER_HEURISTIC",
        "policy": "MlpPolicy",
        "training_seed": training_config.seed,
        "training_config": asdict(training_config),
        "environment_config": asdict(env_config),
        "operator_names": list(OPERATOR_NAMES),
        "operator_contract": dict(operator_contract),
        "model_path": str(model_path),
        "training_seconds": float(training_seconds),
        "phase5_seed_policy": {
            "training_randomness": "separate from evaluation seeds",
            "development_evaluation": "40000-40999",
            "hyperheuristic_validation": "must be predeclared before use",
            "final": "42000-42099 remains embargoed",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "project": package_version("dynamic-manufacturing-digital-twin-rl"),
            "stable_baselines3": package_version("stable-baselines3"),
            "torch": package_version("torch"),
            "gymnasium": package_version("gymnasium"),
        },
        "git": {
            "sha": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
    }


def train_hyperheuristic_ppo(
    output: str | Path,
    *,
    training_config: FJSPHyperHeuristicPPOConfig | None = None,
    env_config: FJSPEnvConfig | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for hyper-heuristic PPO") from exc

    training = training_config or FJSPHyperHeuristicPPOConfig()
    training.validate()
    environment = env_config or FJSPEnvConfig()
    environment.validate()

    contract_env = FlexibleJobShopHyperHeuristicEnv(environment)
    operator_contract = validate_operator_contract(contract_env, seed=training.seed)
    contract_env.close()

    env = FlexibleJobShopHyperHeuristicEnv(environment)
    model = PPO("MlpPolicy", env, **training.algorithm_kwargs())
    started = perf_counter()
    model.learn(total_timesteps=training.total_timesteps)
    training_seconds = perf_counter() - started

    archive = model_archive_path(output)
    archive.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(archive.with_suffix("")))
    env.close()

    manifest_path = (
        Path(metadata_path)
        if metadata_path
        else archive.with_name(f"{archive.stem}_manifest.json")
    )
    manifest = build_training_manifest(
        training_config=training,
        env_config=environment,
        model_path=archive,
        training_seconds=training_seconds,
        operator_contract=operator_contract,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO as an FJSP dispatch hyper-heuristic.")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=801)
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
    parser.add_argument("--output", default="models/fjsp_hyperheuristic_ppo")
    parser.add_argument("--metadata")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training = FJSPHyperHeuristicPPOConfig(
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
    archive, manifest = train_hyperheuristic_ppo(
        args.output,
        training_config=training,
        env_config=environment,
        metadata_path=args.metadata,
    )
    print(f"Hyper-heuristic PPO model saved to {archive}")
    print(f"Training manifest saved to {manifest}")


if __name__ == "__main__":
    main()
