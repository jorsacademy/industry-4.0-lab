# Multi-Training-Seed PPO Validation Protocol

## Purpose

A single PPO training run is not sufficient evidence because policy quality can depend materially on initialization, minibatch order, and stochastic trajectory history. The project therefore treats the training seed as an experimental factor rather than reporting one convenient run.

The validation campaign is predeclared in [`configs/ppo_validation_campaign.json`](../configs/ppo_validation_campaign.json) and executed by the `PPO Validation` GitHub Actions workflow.

## Training design

Five independent PPO members are trained under identical hyperparameters:

- training seeds: `101, 202, 303, 404, 505`;
- total timesteps per member: `150,000`;
- policy: Stable-Baselines3 PPO with `MlpPolicy`;
- learning rate: `3e-4`;
- rollout length: `1024`;
- batch size: `128`;
- gamma: `0.99`;
- GAE lambda: `0.95`;
- entropy coefficient: `0.01`;
- hidden layers: `128 × 128`;
- device: CPU.

Training seeds are below `10000` and are disjoint from validation and final-test seeds.

## Validation regime

Every trained member is evaluated on exactly the same common-random-number validation set:

- validation seeds: `10000–10029`;
- nominal final-test seeds begin at `20000`;
- stress final-test seeds begin at `30000`.

The campaign validator refuses incomplete validation sets or any member whose manifest does not match the predeclared training configuration.

For each member, the workflow retains:

- trained model archive;
- training manifest with runtime/package metadata;
- seed-level validation runs;
- validation summary statistics;
- paired PPO-vs-fixed-rule comparisons.

Model archives and manifests are SHA-256 hashed by the campaign aggregator.

## No best-seed cherry-picking

The research result is not based on the best of five PPO seeds. Final scientific comparisons must retain all five declared training seeds as independent learned-policy realizations.

A single representative model is still useful for dashboards, demonstrations, and later service-layer integration. To avoid selecting the luckiest run, the representative is chosen by a predeclared robust rule:

1. compute each training seed's mean validation priority-weighted tardiness;
2. compute the median of those five means;
3. select the training seed whose WTT mean is closest to that median;
4. break ties by lower measured PPO decision latency;
5. break any remaining tie by lower training seed.

The representative model is therefore a median-role model, not the best validation model.

## Campaign aggregate

`dmdtrl-ppo-campaign` validates all member artifacts and produces:

- `ppo_validation_runs.csv`: PPO-only seed-level validation rows tagged with training seed;
- `ppo_training_seed_summary.csv`: one audited row per independent training run;
- `ppo_validation_manifest.json`: training design, artifact hashes, aggregate training-seed dispersion, and representative-model selection.

Aggregate validation metadata includes the mean, median, sample standard deviation, minimum, and maximum of training-seed WTT means. This quantifies learning instability before final testing.

## Scientific boundary

The validation campaign may be used to verify training stability and freeze the representative model identity. It must not be used to make final claims about PPO superiority over CP-SAT or fixed rules.

After the PPO campaign is frozen, final evaluation will use disjoint `20000+` nominal seeds and `30000+` stress seeds. The CP-SAT controller is already frozen at horizon `8` and `100 ms`; it will not be retuned after observing PPO results.

## Workflow quality gates

The `PPO Validation` workflow fails if:

- fewer than three training seeds are declared;
- training seeds overlap the validation regime;
- validation seeds cross into `20000+` final-test data;
- a model or training manifest is missing;
- a training manifest disagrees with the declared hyperparameters;
- a PPO member does not contain the complete `10000–10029` validation set;
- aggregate campaign files cannot be produced.

The workflow retains member artifacts and aggregate campaign artifacts for 90 days so the validation result can be independently inspected before the final comparative campaign.
