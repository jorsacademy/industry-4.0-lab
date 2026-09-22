# PPO Training and Reproducibility

The PPO pipeline is designed to make a learned-policy result auditable. A model file by itself is not considered a complete experiment output.

## Seed separation

The default experiment contract uses three disjoint ranges:

- training randomness: the explicit PPO training seed and nominal simulator stream;
- nominal out-of-sample evaluation: seeds starting at `20000`;
- distribution-shift evaluation: seeds starting at `30000`.

Final test seeds must not be used for hyperparameter selection.

## Training command

```bash
dmdtrl-train \
  --steps 150000 \
  --seed 42 \
  --output models/ppo_dispatcher \
  --metadata models/ppo_dispatcher_manifest.json \
  --device cpu
```

The command writes:

- `models/ppo_dispatcher.zip` — Stable-Baselines3 PPO model;
- `models/ppo_dispatcher_manifest.json` — training and runtime metadata.

The manifest records:

- PPO hyperparameters;
- complete `EnvConfig`;
- training seed and total timesteps;
- wall-clock training time;
- GitHub commit/ref/run identifiers when available;
- Python/platform details;
- package versions for Stable-Baselines3, PyTorch, Gymnasium, NumPy, and this project;
- nominal/stress evaluation seed conventions.

## Nominal paired evaluation

```bash
dmdtrl-research \
  --model models/ppo_dispatcher.zip \
  --seeds 100 \
  --seed-start 20000 \
  --raw-output results/nominal_runs.csv \
  --summary-output results/nominal_summary.csv \
  --comparisons-output results/nominal_ppo_comparisons.csv
```

Every fixed dispatching policy and PPO receives the same stochastic seed stream.

## Distribution-shift evaluation

```bash
dmdtrl-stress \
  --model models/ppo_dispatcher.zip \
  --seeds 100 \
  --seed-start 30000 \
  --raw-output results/stress_runs.csv \
  --summary-output results/stress_summary.csv \
  --comparisons-output results/stress_ppo_comparisons.csv
```

The PPO model is loaded without retraining. This is intentional: stress tests measure generalization of the nominally trained policy.

## GitHub Actions RL smoke test

`.github/workflows/rl-smoke.yml` is separate from the normal lightweight CI matrix. It installs the RL stack on Python 3.11, trains a short PPO model, runs nominal paired evaluation, runs selected distribution-shift scenarios, validates the generated files, and uploads the complete experiment bundle as a GitHub Actions artifact.

The smoke model is only an integration test. Its short training horizon is not evidence of policy quality and must not be reported as a scientific PPO result.

## Reproducibility limits

A fixed random seed improves reproducibility but does not guarantee bit-for-bit identical neural-network training across all hardware, operating systems, PyTorch versions, or parallel execution settings. For research claims, retain the training manifest and use multiple independently trained agents rather than relying on one seed.

The later full experiment stage should therefore distinguish:

1. **training seeds** — multiple independently trained PPO agents;
2. **evaluation seeds** — common random numbers used to compare policies;
3. **distribution-shift scenarios** — fixed transformations applied only at evaluation time.
