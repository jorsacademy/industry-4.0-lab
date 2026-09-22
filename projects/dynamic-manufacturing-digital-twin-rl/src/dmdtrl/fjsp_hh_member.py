from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from dmdtrl.fjsp_evaluate import instance_fingerprint, summarize_panel
from dmdtrl.fjsp_generator import generate_fjsp_instance
from dmdtrl.fjsp_hh_campaign import (
    PPO_POLICY,
    build_env_config,
    build_generator_config,
    build_training_config,
    validation_seeds,
)
from dmdtrl.fjsp_hh_evaluate import _run_ppo, load_ppo
from dmdtrl.fjsp_hh_protocol import load_hh_validation_design
from dmdtrl.fjsp_hh_train import train_hyperheuristic_ppo
from dmdtrl.fjsp_ppo_campaign import write_csv


def evaluate_member(
    model: Any,
    *,
    design: dict[str, Any],
    training_seed: int,
) -> list[dict[str, float | int | str]]:
    if training_seed not in design["training_seeds"]:
        raise ValueError("training seed is not part of the predeclared campaign")
    generator = build_generator_config(design)
    env_config = build_env_config(design)
    rows: list[dict[str, float | int | str]] = []
    for seed in validation_seeds(design):
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator)
        fingerprint = instance_fingerprint(instance)
        result = _run_ppo(
            model,
            seed=seed,
            expected_fingerprint=fingerprint,
            env_config=env_config,
        )
        rows.append(
            {
                "training_seed": training_seed,
                "seed": seed,
                "seed_regime": "validation",
                "instance_sha256": fingerprint,
                **result,
            }
        )
    return rows


def run_member(
    *,
    config_path: Path,
    training_seed: int,
    output_root: Path,
    bootstrap: int,
) -> tuple[Path, Path, Path, Path]:
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    design = load_hh_validation_design(config_path)
    training = build_training_config(design, training_seed)
    env_config = build_env_config(design)
    output_root.mkdir(parents=True, exist_ok=True)

    model_path, manifest_path = train_hyperheuristic_ppo(
        output_root / "fjsp_hyperheuristic_ppo",
        training_config=training,
        env_config=env_config,
        metadata_path=output_root / "training_manifest.json",
    )
    model = load_ppo(model_path, device=training.device)
    rows = evaluate_member(model, design=design, training_seed=training_seed)
    summary = summarize_panel(rows, bootstrap=bootstrap)
    if len(summary) != 1 or summary[0]["policy"] != PPO_POLICY:
        raise RuntimeError("single-member validation produced an unexpected policy panel")
    summary_row = {"training_seed": training_seed, **summary[0]}
    runs_path = output_root / "ppo_validation_runs.csv"
    summary_path = output_root / "ppo_validation_summary.csv"
    write_csv(rows, runs_path)
    write_csv([summary_row], summary_path)
    return model_path, manifest_path, runs_path, summary_path


def main() -> None:  # pragma: no cover - exercised by GitHub Actions
    parser = argparse.ArgumentParser(
        description="Train and validate one predeclared FJSP PPO hyper-heuristic member."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    args = parser.parse_args()
    model, manifest, runs, summary = run_member(
        config_path=args.config,
        training_seed=args.training_seed,
        output_root=args.output_root,
        bootstrap=args.bootstrap,
    )
    print(f"Model: {model}")
    print(f"Manifest: {manifest}")
    print(f"Validation runs: {runs}")
    print(f"Validation summary: {summary}")


if __name__ == "__main__":
    main()
