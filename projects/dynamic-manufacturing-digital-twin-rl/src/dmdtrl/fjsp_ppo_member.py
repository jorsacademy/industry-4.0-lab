from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dmdtrl.fjsp_env import FJSPEnvConfig
from dmdtrl.fjsp_evaluate import (
    _run_maskable_model,
    instance_fingerprint,
    load_maskable_ppo,
    phase5_seed_regime,
    summarize_panel,
    write_csv,
)
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance

VALIDATION_SEED_MIN = 41_100
FINAL_SEED_MIN = 42_000


def evaluate_member(
    model,
    *,
    training_seed: int,
    seed_start: int,
    seed_count: int,
    generator_config: FJSPGeneratorConfig,
    env_config: FJSPEnvConfig,
) -> list[dict[str, float | int | str]]:
    if training_seed < 0 or training_seed >= 40_000:
        raise ValueError("training_seed must remain below Phase-5 evaluation seeds")
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if seed_start < VALIDATION_SEED_MIN:
        raise ValueError("Maskable PPO validation must start at or above seed 41100")
    if seed_start + seed_count > FINAL_SEED_MIN:
        raise ValueError("Maskable PPO validation seeds must remain below final seed 42000")
    generator_config.validate()
    env_config.validate()
    if env_config.generator != generator_config:
        raise ValueError("environment and generator configurations must match")

    rows: list[dict[str, float | int | str]] = []
    for seed in range(seed_start, seed_start + seed_count):
        instance = generate_fjsp_instance(np.random.default_rng(seed), generator_config)
        fingerprint = instance_fingerprint(instance)
        result = _run_maskable_model(
            model,
            seed=seed,
            expected_fingerprint=fingerprint,
            env_config=env_config,
        )
        rows.append(
            {
                "training_seed": training_seed,
                "seed": seed,
                "seed_regime": phase5_seed_regime(seed),
                "instance_sha256": fingerprint,
                **result,
            }
        )
    return rows


def main() -> None:  # pragma: no cover - exercised by GitHub Actions
    parser = argparse.ArgumentParser(
        description="Evaluate one Phase-5 Maskable PPO training realization on validation seeds."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=41_100)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--operations-min", type=int, default=2)
    parser.add_argument("--operations-max", type=int, default=4)
    parser.add_argument("--eligible-max", type=int, default=3)
    parser.add_argument("--setup-time", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    generator = FJSPGeneratorConfig(
        n_jobs=args.jobs,
        n_machines=args.machines,
        n_families=4,
        operations_min=args.operations_min,
        operations_max=args.operations_max,
        eligible_machines_min=1,
        eligible_machines_max=min(args.eligible_max, args.machines),
    )
    env_config = FJSPEnvConfig(
        generator=generator,
        default_setup_time=args.setup_time,
    )
    model = load_maskable_ppo(args.model, device=args.device)
    rows = evaluate_member(
        model,
        training_seed=args.training_seed,
        seed_start=args.seed_start,
        seed_count=args.seeds,
        generator_config=generator,
        env_config=env_config,
    )
    summary = summarize_panel(rows, bootstrap=args.bootstrap)
    if len(summary) != 1 or summary[0]["policy"] != "MASKABLE_PPO":
        raise RuntimeError("single-member validation produced an unexpected policy panel")
    summary_row = {"training_seed": args.training_seed, **summary[0]}
    write_csv(args.raw_output, rows)
    write_csv(args.summary_output, [summary_row])
    print(
        f"Maskable PPO seed {args.training_seed}: "
        f"WTT={float(summary_row['weighted_tardiness_mean']):.3f} "
        f"latency={float(summary_row['mean_decision_time_ms']):.3f} ms"
    )


if __name__ == "__main__":
    main()
