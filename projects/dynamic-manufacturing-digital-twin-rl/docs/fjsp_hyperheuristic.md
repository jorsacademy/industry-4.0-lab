# Phase-5 FJSP PPO hyper-heuristic redesign

The first Phase-5 learned controller used a large masked action space over concrete
`(job, operation, machine)` assignments. Five independent 150,000-step Maskable PPO
training realizations were all materially worse than the frozen CP-SAT, EDD, and SPT
comparators on the separate PPO-validation block. That result is frozen in
`configs/fjsp_direct_ppo_validation_freeze.json` and is not overwritten by this redesign.

## Research change

The next learned controller changes the action abstraction rather than tuning the
rejected flat policy after observing validation results.

At each simulator decision epoch PPO selects one of eight deterministic dispatch
operators:

1. `EARLIEST_DUE_DATE`;
2. `SHORTEST_PROCESSING`;
3. `MINIMUM_SETUP`;
4. `HIGHEST_PRIORITY`;
5. `MINIMUM_SLACK`;
6. `CRITICAL_RATIO`;
7. `SAME_FAMILY_FIRST`;
8. `WEIGHTED_TARDINESS_RISK`.

The selected operator then maps the current simulator state to exactly one concrete
precedence- and resource-feasible FJSP assignment. PPO never emits a job, operation,
or machine identifier directly.

## Why this is a different hypothesis

The direct policy had to learn both dispatch logic and combinatorial feasibility in a
large instance-indexed action space. The hyper-heuristic instead learns a sequential
operator-selection policy over a small fixed action space. Domain rules perform the
last-mile feasible assignment.

This tests a more defensible decision-intelligence question: can an adaptive policy
choose among individually interpretable scheduling operators better than committing
to one fixed rule under changing shop states?

## Transition equivalence

`FlexibleJobShopHyperHeuristicEnv` delegates instance generation, observations,
simulator transitions, reward calculation, and KPI accounting to the existing
`FlexibleJobShopEnv`. The new layer changes only the action abstraction.

Contract tests reset both environments on the same instance seed, select the same
underlying EDD assignment, and require identical next observations, rewards, and
scheduled operations.

## Action feasibility

All eight operators select only from `FlexibleJobShopSimulator.eligible_actions()`.
Therefore every operator is admissible at every non-terminal decision epoch. The
operator action space is fixed at eight actions, independent of job count, operation
count, or machine count.

## Evidence boundary

This core PR does not train or select a new PPO model. It introduces only the operator
library, the PPO-ready environment wrapper, and deterministic tests.

The direct-policy final block `42000-42099` remains embargoed. Hyper-heuristic
development and smoke work must stay inside the already designated development block
`40000-40999`. A new validation block must be predeclared before multi-seed model
selection, and no Phase-5 final seed may be accessed merely because the previous RL
architecture failed.
