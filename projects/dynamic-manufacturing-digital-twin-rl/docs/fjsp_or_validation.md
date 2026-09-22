# Phase 5 FJSP CP-SAT Validation and Freeze Protocol

## Scope

The Phase-5 FJSP rolling-horizon CP-SAT controller must be frozen before learned-policy validation. This prevents solver horizon or compute budget from being adjusted after observing Maskable PPO validation or final outcomes.

## Validation seeds

The operating-point campaign uses exactly 30 common validation instances:

`41000-41029`

These seeds are inside the Phase-5 validation partition and are disjoint from:

- development/smoke: `40000-40999`;
- Phase-5 final: `42000+`;
- all locked Phase-4 seeds.

## Grid

The predeclared grid is:

- job horizons: `4, 8, 12` released jobs;
- online solve budgets: `20, 50, 100 ms`;
- total configurations: 9;
- total raw episodes: `9 x 30 = 270`.

Every configuration uses the same generator distribution:

- 12 jobs;
- 5 machines;
- 2-4 operations per job;
- 1-3 eligible machines per operation;
- 4 families;
- setup time 1.0 for family changes unless a richer matrix is introduced in a future model revision.

## Common-instance identity

Every raw row contains the canonical FJSP instance SHA-256. Validation fails if:

- any grid configuration is missing a declared seed;
- any configuration contains duplicate seeds;
- a raw row is not labeled `validation`;
- the same seed produces different instance fingerprints across configurations.

## Quality and compute objectives

The sensitivity table reports:

- mean weighted tardiness and bootstrap interval;
- mean decision latency and bootstrap interval;
- solver fallback rate;
- solver success rate.

Pareto dominance is evaluated jointly on lower weighted tardiness and lower mean decision latency.

## Reliability-gated selection rule

The operating point is selected without access to learned-policy validation results:

1. keep Pareto-optimal configurations;
2. require mean solver fallback rate <= 1%;
3. identify the best WTT among reliable Pareto points;
4. allow points within 2% of that best WTT;
5. select the point with the lowest measured mean decision latency;
6. break ties by WTT, fallback rate, solver budget, horizon, then policy name.

This makes the selected controller an operational WTT/compute compromise rather than simply choosing the most expensive solver configuration.

## Statistical reference comparisons

Every non-reference grid point is paired against `H=12 / 100 ms` on the same 30 instances for:

- weighted tardiness;
- online decision latency.

The comparison reports bootstrap paired intervals, paired sign-flip randomization p-values, paired Cohen `dz`, and probability of superiority.

These comparisons explain the trade-off surface; they do not override the predeclared selection rule.

## Freeze artifact

The workflow writes:

- `fjsp_or_validation_runs.csv`;
- `fjsp_or_validation_summary.csv`;
- `fjsp_or_validation_comparisons.csv`;
- `fjsp_cpsat_selection.json`.

The selection JSON records the validation seed boundary, Git SHA, selected horizon/budget, WTT, measured latency, fallback rate, quality tolerance, and reliability threshold.

After validation succeeds, the selected operating point will be committed as a permanent Phase-5 configuration in a separate freeze PR. The 30-seed campaign will not be rerun merely to obtain a more favorable point.

## Next gate

Only after the FJSP CP-SAT operating point is frozen will the repository launch the multi-training-seed Maskable PPO validation campaign on the same Phase-5 validation partition. Final seeds remain inaccessible until both controller families are frozen.
