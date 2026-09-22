# PPO Multi-Seed Validation Results

## Campaign status

The predeclared five-training-seed PPO validation campaign completed successfully on GitHub Actions run `33210409297`.

The campaign trained five independent policies for 150,000 timesteps each (750,000 total training timesteps), evaluated every policy on the same validation seeds `10000–10029`, verified training manifests and seed boundaries, and aggregated the member artifacts without using nominal final-test (`20000+`) or stress final-test (`30000+`) data.

## Training-seed results

| Training seed | Validation mean weighted tardiness | Mean PPO decision latency (ms) | Training time (s) |
| ---: | ---: | ---: | ---: |
| 101 | 77.0644 | 0.1922 | 85.66 |
| 202 | 83.4609 | 0.2469 | 110.80 |
| 303 | 73.8590 | 0.2763 | 121.50 |
| 404 | 72.9942 | 0.1911 | 85.51 |
| 505 | 83.2213 | 0.1036 | 54.94 |

Across training seeds:

- mean validation WTT: **78.1200**;
- median validation WTT: **77.0644**;
- sample standard deviation across training-seed WTT means: **5.0023**;
- minimum: **72.9942**;
- maximum: **83.4609**;
- mean PPO decision latency across training-seed summaries: **0.2020 ms**.

The spread is material enough that a single PPO training run would not be an adequate basis for a scientific claim.

## Fixed-rule context

The strongest fixed rule on the same nominal validation seeds was `WEIGHTED_COMPOSITE`, with mean WTT **73.8590**.

The PPO training seeds therefore behaved differently:

- seed 404 was descriptively better than the weighted composite mean;
- seed 303 was effectively equal at the displayed precision;
- seeds 101, 202, and 505 were worse on mean WTT.

This is a central result of the validation stage. PPO is not being presented as automatically superior because it is learned. Training-seed variation changes the conclusion if one cherry-picks a run.

The frozen CP-SAT validation campaign produced mean WTT **72.5101** at H=8 / 100 ms on the same validation seed range. That value is useful model-selection context, but final PPO-vs-CP-SAT claims are deferred to the disjoint final-test campaign with direct paired analysis.

## Representative model

The predeclared representative-model rule selects the training run whose validation WTT mean is closest to the median across all five training seeds. The median is 77.0644, so the representative model is:

- training seed: **101**;
- validation WTT: **77.0644**;
- model SHA-256: `a3172d12c59a8585a2ded6ff8e1ae2bbf3287b5ca97d183cf06841e12d6980e3`.

The representative model is **not** the best PPO seed. Seed 404 has the lowest validation WTT. Seed 101 is frozen only for dashboard/demo/service continuity because it is the median-role realization.

Final research results must retain all five training-seed realizations.

## Artifact verification

The aggregate campaign artifact:

- artifact ID: `9701517564`;
- archive SHA-256: `dc20c0bbc6b083e048edfb97e9a85e34a4308b885e6e614faa338089669e2d07`.

The aggregate ZIP was downloaded independently after the workflow and its SHA-256 digest matched the GitHub Actions digest exactly. Its manifest reproduced the independently calculated mean, median, standard deviation, and representative seed.

Per-member model hashes and workflow artifact digests are frozen in [`../configs/ppo_validation_freeze.json`](../configs/ppo_validation_freeze.json).

## Interpretation boundary

This validation stage supports three conclusions only:

1. the PPO training/evaluation pipeline is reproducible and auditable;
2. PPO performance has non-negligible training-seed variability under the declared setup;
3. seed 101 is the predeclared median-role representative for non-scientific deployment/demo use.

It does **not** establish final superiority or inferiority of PPO relative to CP-SAT or fixed rules. Those claims require disjoint nominal and stress final-test seeds after both controller families are frozen.
