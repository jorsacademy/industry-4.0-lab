# Phase-5 Maskable PPO validation

This stage tested whether a mask-aware policy that directly selects `(job, operation, machine)` actions is robust enough to justify access to the Phase-5 final-test instances.

## Frozen comparison boundary

The rolling-horizon CP-SAT baseline was selected first, independently of PPO, on seeds `41000-41029`. Its frozen operating point is job horizon `4`, online solve budget `100 ms`, one search worker, and fixed solver random seed. The exact selection provenance is stored in `configs/fjsp_cpsat_validation_freeze.json`.

## Seed separation

| Role | Seeds |
| --- | --- |
| development/smoke | `40000-40999` |
| CP-SAT tuning | `41000-41029` |
| direct Maskable PPO validation | `41100-41129` |
| nominal final test | `42000-42099` |

The validation workflow asserted that no evaluation row reached seed `42000`.

## Training design

Five independent Maskable PPO realizations were trained with seeds `701, 1701, 2701, 3701, 4701`. Every member used the same predeclared configuration: 150,000 timesteps, learning rate `3e-4`, rollout length `1024`, batch size `128`, gamma `0.99`, GAE lambda `0.95`, entropy coefficient `0.01`, two 128-unit hidden layers, and CPU execution.

The FJSP generator stayed fixed at 12 jobs, 5 machines, 2-4 operations per job, and up to 3 eligible machines per operation.

## Statistical unit

The 150 PPO validation episodes were not pooled as independent algorithm replications. For each training seed, paired WTT differences were first computed over the same 30 canonical instance fingerprints. The top-level robustness summary then used the five independent training-seed realizations as its inference unit.

## Validation result

The direct-action policy failed the validation gate.

| Controller | Validation WTT mean |
| --- | ---: |
| rolling-horizon CP-SAT | 20.416 |
| earliest due date | 33.587 |
| shortest processing | 52.943 |
| direct Maskable PPO, mean over five training seeds | 82.512 |

The five PPO member WTT means were `84.519`, `82.917`, `82.823`, `82.948`, and `79.352`. Mean decision latency was approximately `0.370 ms`, so the learned dispatcher was computationally fast but operationally poor.

At the independent training-seed level, mean paired WTT improvement was negative against every comparator:

- vs EDD: `-48.925`, bootstrap CI `[-50.279, -47.191]`;
- vs frozen CP-SAT: `-62.096`, bootstrap CI `[-63.450, -60.362]`;
- vs SPT: `-29.568`, bootstrap CI `[-30.916, -27.841]`.

Training-seed win fraction was `0/5` against all three baselines. The representative median-role member is seed `1701`; it is retained only for reproducibility and demo continuity, not as a positive performance result.

The full frozen result and model hashes are stored in `configs/fjsp_direct_ppo_validation_freeze.json`.

## Decision gate

The final nominal block `42000-42099` remains embargoed. Spending held-out final instances on a controller that failed all validation baselines would add little evidence and would consume the clean final-test budget.

The next RL design changes the action abstraction rather than tuning the rejected flat policy. PPO will act as a small fixed-action hyper-heuristic that selects among strong feasible dispatching operators. The redesigned controller must pass a fresh validation gate before any Phase-5 final seed is accessed.
