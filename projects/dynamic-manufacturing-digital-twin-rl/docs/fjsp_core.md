# Phase 5: Flexible Job-Shop Core

## Purpose

Phase 4 showed that the PPO hyper-heuristic did not add robust value over the strongest fixed dispatching rule in the dynamic parallel-machine formulation. Phase 5 therefore changes the scheduling problem structure instead of tuning PPO against locked final-test seeds.

Phase 5 is implemented as a separate true flexible job-shop (FJSP) stack so the validated v1.0 parallel-machine experiment remains reproducible and frozen.

## Implemented FJSP structure

The FJSP model supports:

- jobs with multiple ordered operations;
- strict operation precedence;
- alternative eligible machines for each operation;
- machine-dependent processing times for each operation;
- dynamic job release times;
- family-dependent sequence setup times;
- event-driven advancement to the next feasible decision epoch;
- explicit `(job, operation, machine)` assignment actions;
- operation-level schedules and job-level tardiness/flow-time metrics.

`FJSPAction(job_id, operation_index, machine_id)` is feasible only when:

1. the job has been released;
2. the requested operation is exactly the job's next unscheduled operation;
3. its predecessor has completed;
4. the selected machine is eligible for that operation; and
5. the machine is available at the current event time.

The simulator advances time only when no feasible assignment exists. This creates a dynamic feasible action set naturally from precedence, routing eligibility, releases, and machine availability.

## Reproducible instance generator

`generate_fjsp_instance` creates seeded stochastic instances with configurable:

- number of jobs and machines;
- operations per job;
- eligible machines per operation;
- release process;
- processing-time range;
- due-date tightness;
- priorities and product families.

Due dates are based on the sum of the fastest eligible processing option for each operation, multiplied by a stochastic due-date factor. This is intentionally simple and will be stress-tested before final experimental use.

## Deterministic baselines

The FJSP core currently includes two feasibility-preserving direct assignment selectors:

- shortest setup-plus-processing action;
- earliest-due-date action with processing-time tie breaking.

These are scaffolding baselines, not the final strong comparison set. Later Phase-5 increments will add FJSP rolling-horizon CP-SAT / interval scheduling and neighborhood-search baselines before any RL performance claim.

## Gymnasium environment and action mask

`FlexibleJobShopEnv` wraps the event-driven FJSP simulator in a fixed-capacity Gymnasium interface.

The action space is a deterministic index over all capacity slots:

`job slot × operation slot × machine`.

The environment exposes `action_masks()`, a Boolean vector with one entry per discrete action. An action is true only when the corresponding `(job, operation, machine)` tuple is currently precedence-feasible, released, machine-eligible, and resource-available.

This separates two concerns:

- the Gymnasium action space remains fixed across an experiment;
- the feasible action set remains dynamic and exact.

Invalid unmasked actions are rejected rather than silently repaired. A future Maskable PPO integration will consume the same mask so the learned policy never trains on impossible assignments.

### Observation structure

The current fixed-size observation contains:

- global time/progress/utilization/slack/overdue features;
- per-job release/completion/progress/priority/family/slack/readiness features;
- per-job next-operation eligible-machine indicators;
- per-job next-operation machine-dependent normalized processing times;
- per-machine availability, time-to-available, previous family, busy load, and setup load.

All features are normalized/clipped to `[0, 1]`. This vector representation is intentionally explicit and auditable. Graph/attention encoders are deferred until a strong vector baseline exists.

### Action trace

Every executed decision records:

- decision index and simulated decision time;
- encoded action id;
- job, operation, and machine ids;
- feasible-action count at the decision epoch;
- whether the operation completed the job;
- scheduled completion time;
- immediate reward.

This directly addresses the Phase-4 diagnostic gap where some PPO models were outcome-equivalent to deterministic heuristics but action traces were unavailable.

## Reward contract

The initial FJSP reward is operational and incremental:

- small operation-completion bonus;
- additional final-job completion bonus;
- operation waiting penalty;
- sequence setup penalty;
- priority-weighted tardiness penalty applied when the final operation is scheduled.

This reward is not frozen for scientific comparison yet. It remains a development parameter until strong FJSP OR/search baselines and Phase-5 validation partitions are established.

## Testing contract

The tests verify:

- invalid precedence chains are rejected;
- duplicate machine alternatives are rejected;
- seeded FJSP generation is reproducible;
- operation precedence cannot be bypassed;
- machine eligibility is enforced;
- release times and event advancement work;
- sequence setup time is applied between product families;
- a deterministic greedy policy can complete a generated multi-operation FJSP instance;
- action codec encode/decode is stable;
- the Boolean mask exactly matches simulator-feasible actions;
- precedence remains enforced after a masked step;
- infeasible discrete actions are rejected;
- action-trace records preserve the executed decision contract.

## Deliberate exclusions at this stage

Not yet included:

- breakdowns or stochastic repairs in the FJSP stack;
- urgent-order burst process;
- pre-generated exogenous disruption plans;
- `sb3-contrib` / Maskable PPO training;
- GNN or attention encoder;
- FJSP CP-SAT benchmark;
- physical completion-event queue separate from schedule commitment.

## Next Phase-5 increment

The next increment should add the first strong FJSP optimization baseline: rolling-horizon CP-SAT with operation precedence, alternative-machine intervals, and sequence/setup logic. That baseline should be validated before Maskable PPO is trained so RL is not developed without a credible OR comparator.

The Phase-4 final seeds remain locked and are not reused for Phase-5 model selection.
