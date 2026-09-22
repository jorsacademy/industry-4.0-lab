# Phase 5 FJSP Rolling-Horizon CP-SAT Baseline

## Purpose

The masked FJSP environment now has a strong optimization comparator before Maskable PPO is introduced.

The controller is a receding-horizon CP-SAT policy over the same `FlexibleJobShopSimulator` state used by deterministic and future learned controllers. It executes exactly one currently feasible `(job, operation, machine)` action, then replans at the next event epoch.

## Information boundary

The optimizer is deliberately online and non-oracular.

At decision time `t` it can use:

- jobs with `release_time <= t`;
- the complete remaining operation route of each released job;
- alternative eligible machines and processing times for those released jobs;
- current machine availability and last processed family;
- already committed predecessor completion times.

It cannot use unreleased jobs or their future arrival times when forming the candidate horizon.

## Candidate horizon

Released unfinished jobs are ranked by an urgency key based on priority-adjusted due-date slack, due date, priority, and job id. The first `job_horizon` jobs are modeled in full from their next unscheduled operation through the final operation.

The initial development default is eight released jobs. This is not yet a frozen scientific operating point.

## CP-SAT formulation

For every remaining operation in a candidate job:

- exactly one eligible machine is selected;
- integer begin/completion times are created on a configurable time scale;
- job precedence enforces operation `k+1` after operation `k`;
- the first unscheduled operation respects its already-known ready time.

For each machine, `AddCircuit` creates an explicit route through the operations assigned to that machine. Self loops deactivate operations assigned elsewhere. Depot arcs represent the first and last operations after the machine's current committed state.

Sequence-dependent setup is tied to the active predecessor arc:

- depot -> operation uses the machine's current `last_family`;
- operation A -> operation B uses the family transition `family(A) -> family(B)`.

Completion time therefore includes both processing and the setup implied by the active machine predecessor.

## Receding-horizon action contract

The model contains one Boolean `first_choice` variable for every simulator-feasible action at the current event epoch.

Exactly one must be selected. A selected action:

- must be assigned to the selected machine;
- must be the first modeled operation on that machine after the current machine state;
- must begin setup at the current simulator time.

Only that action is executed in the simulator. All other CP-SAT decisions are discarded and recomputed at the next event epoch.

## Objective

Priority-weighted tardiness is lexicographically dominant.

The integer objective is:

`primary_weight * total_priority_weighted_tardiness + makespan + total_setup`

where `primary_weight` is constructed above an upper bound on the complete secondary term. Therefore the optimizer cannot trade away one scaled unit of weighted tardiness merely to reduce setup or makespan.

## Reliability and latency

The solver uses:

- one search worker;
- a fixed CP-SAT random seed;
- a bounded per-decision solve budget.

`OPTIMAL` and `FEASIBLE` produce an OR action. `UNKNOWN` caused by a short online budget uses a deterministic earliest-due-date feasible fallback and records the fallback. `INFEASIBLE` and model-invalid states remain hard failures.

Reported controller diagnostics include:

- decision count;
- fallback count/rate;
- solver success rate;
- mean solver wall time in milliseconds.

## Development benchmark

`python -m dmdtrl.fjsp_or_benchmark` evaluates three controllers on identical seeded FJSP instances:

- shortest setup-plus-processing;
- earliest due date;
- rolling-horizon CP-SAT.

The raw table contains weighted tardiness, makespan, flow time, setup, utilization, latency and solver-reliability fields.

This first benchmark is for integration and model development, not a final performance claim.

## Phase-5 seed partition

Phase 5 does not reuse the locked Phase-4 final seeds.

- development/smoke: `40000–40999`;
- validation/model selection: `41000–41999`;
- final evaluation: `42000+` (exact final range will be frozen before access).

The CP-SAT horizon and solve budget will be selected only from the Phase-5 validation range before Maskable PPO final comparison.

## Next step

After this OR baseline is stable, the next algorithmic increment is `sb3-contrib` Maskable PPO using the already-tested `action_masks()` contract. RL training will be compared against the OR controller and deterministic FJSP baselines on common Phase-5 instances; best-seed reporting will not replace multi-training-seed evaluation.
