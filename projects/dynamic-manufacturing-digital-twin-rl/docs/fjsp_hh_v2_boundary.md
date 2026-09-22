# FJSP hyper-heuristic v2 validation boundary

## Why v2 exists

The predeclared `operator_selection_v1` validation campaign did not justify promotion to the Phase-5 final benchmark. Across five independent PPO training seeds, v1 underperformed both the strongest fixed dispatch rules and the frozen rolling-horizon CP-SAT controller on weighted tardiness. The final block therefore remains untouched.

The v2 architecture hypothesis is narrower than “train PPO longer.” The fixed eight-operator action abstraction remains intact because it guarantees feasible decisions and low online latency. The proposed change is informational: the controller should observe attributes of the concrete feasible action currently proposed by each operator, in addition to the existing global FJSP state.

## Reserved validation block

`41300-41329` is reserved as the independent 30-instance validation block for `operator_selection_v2`.

Reservation does **not** authorize access. No v2 implementation, diagnostic, training run, hyperparameter search, model selection, or smoke workflow may read these seeds until a later exact v2 validation protocol has been merged to `main`.

The data boundary is:

| Purpose | Seed block | Status |
| --- | --- | --- |
| Development / architecture diagnostics | `40000-40999` | usable |
| v1 validation | `41200-41229` | consumed; historical evidence only |
| v2 validation | `41300-41329` | reserved and embargoed |
| Phase-5 final nominal | `42000-42099` | embargoed |

## Architecture direction

The v2 observation family is `global_plus_operator_conditioned_candidate_features`. The action space remains the same eight frozen operators.

Development work may investigate features derived from each operator's currently proposed feasible assignment, including:

- job slack;
- processing duration on the proposed machine;
- setup duration;
- machine readiness / earliest-start information;
- job priority;
- critical ratio;
- projected weighted-tardiness risk.

These are feature families, not yet the exact frozen tensor schema. Their normalization, ordering, dimensionality, redundancy handling, and any additional development-only diagnostic features may be revised using only the development block.

## Required gate before v2 validation

A separate protocol PR must be merged before `41300-41329` can be accessed. That protocol must freeze at least:

- the exact observation schema and normalization;
- independent PPO training seeds;
- PPO hyperparameters and training budget;
- the complete comparator set, including all eight fixed operators and the already-frozen `FJSP_CPSAT_H4_B100MS` controller;
- the statistical aggregation rule at the independent training-seed level.

No hyperparameter search is permitted on the v2 validation block. The Phase-5 final block cannot be opened unless v2 passes the subsequent validation promotion gate.

The machine-readable reservation is `configs/fjsp_hh_v2_boundary.json`.
