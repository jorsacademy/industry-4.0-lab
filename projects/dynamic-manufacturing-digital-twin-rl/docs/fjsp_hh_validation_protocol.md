# Phase-5 FJSP hyper-heuristic validation protocol

This protocol is committed before the multi-training-seed hyper-heuristic campaign is implemented or executed.

## Hypothesis under test

The rejected Phase-5 direct-action controller asked Maskable PPO to select concrete `(job, operation, machine)` assignments. The replacement controller asks plain PPO to select one of eight deterministic dispatch operators while the operator performs the final feasible assignment.

The validation question is whether this smaller, interpretable action abstraction produces a robust adaptive policy that improves weighted tardiness over fixed dispatch rules and is competitive with the already-frozen rolling-horizon CP-SAT comparator, after accounting for online decision latency.

## Frozen action abstraction

The action IDs and operator semantics are fixed before training:

1. earliest due date;
2. shortest processing;
3. minimum setup;
4. highest priority;
5. minimum slack;
6. critical ratio;
7. same-family-first;
8. weighted-tardiness-risk.

All eight operators remain fixed baselines as well as PPO actions. The campaign therefore tests adaptive operator selection against committing to any one member of the same operator library.

## Training design

Five independent PPO training realizations are declared:

`901, 1901, 2901, 3901, 4901`

Each realization uses the same fixed configuration:

- 150,000 timesteps;
- learning rate `3e-4`;
- rollout length `1024`;
- batch size `128`;
- gamma `0.99`;
- GAE lambda `0.95`;
- entropy coefficient `0.01`;
- two 128-unit policy/value hidden layers;
- CPU execution.

These choices deliberately match the core PPO budget used for the rejected direct-action controller. The two-seed development smoke is not used to tune them. This keeps the next comparison focused on action abstraction rather than post-hoc hyperparameter search.

## Seed separation

| Role | Seeds |
| --- | --- |
| development / smoke | `40000-40999` |
| frozen CP-SAT tuning | `41000-41029` |
| rejected direct-action PPO validation | `41100-41129` |
| hyper-heuristic PPO validation | `41200-41229` |
| Phase-5 final nominal test | `42000-42099` |

The new campaign may use `41200-41229` only after this protocol is merged. It must not reuse the direct-action validation instances and must not access final seeds.

## Baselines

The comparison panel is frozen to:

- all eight fixed dispatch operators;
- rolling-horizon CP-SAT at the previously selected `H=4`, `100 ms`, one-worker operating point.

CP-SAT is not retuned during the PPO campaign.

## Statistical unit

For each PPO training seed, all controllers are evaluated on the same 30 canonical validation instances. Instance SHA-256 fingerprints must match across policies.

Paired weighted-tardiness differences are first computed within each training realization across the 30 common instances. The algorithm-level robustness summary then uses the five independent training realizations as the top-level inference unit. The 150 PPO validation episodes are not treated as 150 independent algorithm replications.

The campaign must retain:

- WTT, makespan, flow time and decision latency;
- per-instance common-random-number fingerprints;
- per-training-seed comparisons against every fixed operator and CP-SAT;
- across-training-seed bootstrap summaries;
- model and training-manifest hashes;
- training duration and runtime package versions;
- operator-selection diversity diagnostics.

## Model reporting rule

No best-seed cherry-picking is allowed. All five training seeds must remain in scientific claims.

A representative model may be selected only for demo/deployment continuity: choose the training realization whose validation WTT is closest to the median across all five realizations, then break ties by lower inference latency and lower training seed.

## Final-test embargo

The Phase-5 final block `42000-42099` remains unavailable during this campaign. A favorable or unfavorable hyper-heuristic validation result does not justify inspecting final instances before the validation result itself has been frozen and the next gate explicitly declared.

The machine-readable source of truth is `configs/fjsp_hh_validation_design.json`; `dmdtrl.fjsp_hh_protocol` validates its frozen boundaries before any future campaign execution.
