# Phase 5 Maskable PPO Integration

## Purpose

The true FJSP stack has a fixed-capacity discrete action space whose feasible subset changes with job release, operation precedence, routing eligibility, and machine availability. Standard PPO would waste probability mass on impossible assignments and could attempt invalid actions.

Phase 5 therefore uses `sb3-contrib` Maskable PPO, which applies the environment's Boolean `action_masks()` during rollout collection and prediction.

This integration is intentionally added only after the FJSP rolling-horizon CP-SAT comparator exists.

## Dependency contract

The integration targets the current compatible release line:

- `sb3-contrib >= 2.9, < 3`;
- Stable-Baselines3 is resolved by `sb3-contrib` and must be on the compatible 2.9+ line;
- Python 3.10+.

The Phase-5 smoke workflow installs this dependency directly rather than changing `pyproject.toml`, because the legacy Phase-3 PPO validation workflow intentionally treats `pyproject.toml` as a retraining trigger. Avoiding that file prevents the frozen 750,000-timestep legacy campaign from being rerun for an unrelated Phase-5 dependency addition.

## Mask contract

`FlexibleJobShopEnv.action_masks()` returns one Boolean per encoded `job x operation x machine` action slot:

- `True`: the assignment is currently feasible;
- `False`: the assignment violates release, precedence, routing eligibility, or machine availability.

`validate_mask_contract` runs one complete generated FJSP episode with deterministic valid-mask actions and verifies:

- mask shape exactly equals the discrete action-space size;
- every nonterminal state has at least one feasible action;
- no truncated episode is produced;
- the episode reaches completion.

The standard SB3/Gym random-action environment checker is not used as the mask validator because it does not sample from the invalid-action mask.

## Training configuration

`FJSPMaskablePPOConfig` records:

- total training timesteps;
- training seed;
- learning rate;
- rollout length and minibatch size;
- gamma and GAE lambda;
- entropy coefficient;
- actor/critic hidden width;
- device and logging verbosity.

The first development defaults are not frozen scientific hyperparameters. They exist to establish a reproducible training contract before Phase-5 validation.

## Reproducibility manifest

Every training run writes a JSON manifest containing:

- algorithm and policy type;
- complete training configuration;
- complete FJSP environment/generator configuration;
- model path;
- wall-clock training time;
- mask-contract diagnostics;
- Python/platform metadata;
- project, `sb3-contrib`, Stable-Baselines3, PyTorch, Gymnasium, and NumPy versions;
- GitHub Actions SHA/ref/workflow/run id when available;
- Phase-5 seed partition policy.

## FJSP RL Smoke workflow

The dedicated workflow performs a real external-dependency integration test on Python 3.11:

1. install the project plus `sb3-contrib >= 2.9, < 3`;
2. run FJSP core/env/training contract tests;
3. train a 4,096-timestep Maskable PPO model on a small 6-job/3-machine FJSP distribution;
4. load the saved model;
5. run deterministic mask-aware inference on development seeds `40010` and `40011`;
6. assert every predicted action is feasible under the current mask;
7. retain model, manifest, and compact inference diagnostics as a 14-day artifact.

This workflow is an integration smoke test, not policy-quality evidence.

## Scientific next step

After smoke integration is stable, Phase 5 needs a dedicated comparison/validation harness using common generated instances for:

- deterministic shortest-processing baseline;
- deterministic earliest-due-date baseline;
- rolling-horizon FJSP CP-SAT;
- Maskable PPO.

The first scientific learned-policy campaign will use multiple independent PPO training seeds and Phase-5 validation seeds `41000-41999`. Hyperparameters and the CP-SAT operating point will be frozen before any Phase-5 final seed `42000+` is accessed.

The best PPO training seed will not replace the multi-training-seed result.
