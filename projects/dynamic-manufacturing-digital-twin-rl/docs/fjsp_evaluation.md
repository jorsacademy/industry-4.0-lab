# Phase 5 Common-Seed FJSP Evaluation Protocol

## Goal

Phase 5 evaluates learned, optimization, and deterministic controllers on exactly the same generated flexible job-shop instances. A shared integer seed is not accepted as sufficient evidence by itself: each controller row carries the SHA-256 fingerprint of the canonical generated instance.

## Controller panel

The current evaluator supports:

- `SHORTEST_PROCESSING` — shortest current setup-plus-processing assignment;
- `EARLIEST_DUE_DATE` — earliest job due date with processing-time tie breaking;
- `ROLLING_HORIZON_CPSAT` — released-job-only FJSP CP-SAT replanning;
- `MASKABLE_PPO` — deterministic inference with the exact dynamic feasibility mask.

The learned controller is optional so OR/heuristic development can run without `sb3-contrib` installed.

## Common-random-number contract

For every environment seed:

1. the evaluator creates a canonical `FJSPInstance` with `numpy.random.default_rng(seed)`;
2. the instance is serialized deterministically and SHA-256 hashed;
3. deterministic and CP-SAT controllers operate directly on that canonical instance;
4. the Gymnasium Maskable PPO environment resets with the same seed;
5. the environment-generated instance is independently fingerprinted;
6. evaluation aborts if the two fingerprints differ.

This converts the seed convention into a tested identity contract rather than assuming two RNG pathways are equivalent.

## Seed partitions

- development/smoke: `40000-40999`;
- validation/model selection: `41000-41999`;
- final evaluation: `42000+`, with an exact interval to be frozen before any final access.

Phase-4 final seeds are never reused.

## Raw metrics

Each `(seed, controller)` row contains:

- canonical instance SHA-256;
- weighted tardiness;
- makespan;
- mean flow time;
- setup time;
- utilization;
- mean online decision latency;
- CP-SAT fallback rate and solver-success rate;
- for Maskable PPO, unique encoded-action fraction and mean feasible-action count.

The action diagnostics are not substitutes for direct policy entropy or state-conditioned switching analysis, but they provide an early trace-level check against trivial action collapse.

## Summary and paired inference

Weighted tardiness summaries use a bootstrap confidence interval over common environment seeds.

Candidate comparisons use seed-paired differences. For weighted tardiness:

`improvement = baseline WTT - candidate WTT`

so positive values favor the candidate.

Reported paired fields include:

- mean improvement;
- percentage improvement relative to baseline mean;
- bootstrap confidence interval;
- paired sign-flip randomization p-value;
- paired Cohen `dz` effect size;
- probability of superiority;
- number of paired seeds.

Smoke workflows use deliberately tiny resample counts. Scientific validation uses larger predeclared counts.

## Current smoke purpose

`FJSP RL Smoke` trains a short Maskable PPO model and evaluates it with CP-SAT and deterministic controllers on development seeds only. It checks dependency integration, valid masked inference, common-instance identity, output schemas, and end-to-end artifact generation.

A favorable smoke WTT is not evidence of algorithm superiority.

## Next research gates

Before Phase-5 final comparison:

1. run CP-SAT horizon/solve-budget sensitivity only on validation seeds;
2. freeze a reliable FJSP CP-SAT operating point;
3. run multiple independent Maskable PPO training seeds;
4. compare every learned model on common validation instances;
5. report training-seed dispersion rather than the best learned seed;
6. freeze PPO configuration/model set;
7. freeze the exact `42000+` final seed range;
8. run final paired and hierarchical inference once, without retuning.
