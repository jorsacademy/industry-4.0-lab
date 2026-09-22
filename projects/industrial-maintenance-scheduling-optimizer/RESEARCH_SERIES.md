# Maintenance, Reliability, and Asset-Decision Research Series

This file maps repositories related to maintenance planning, predictive maintenance, condition monitoring, and asset-control decisions. It is an index only: each repository remains independent because prediction, scheduling, control, and sensing are different decision layers.

## Maintenance optimization and scheduling

- `industrial-maintenance-scheduling-optimizer` — preventive/predictive maintenance task generation, priority scoring, resource-constrained scheduling, and schedule evaluation.
- `aircraft-maintenance-scheduling-gurobi` — aviation-specific maintenance scheduling with a different operational constraint system.

## Predictive maintenance and decision support

- `predictive-maintenance-sensor-data-optimization-python` — sensor-data based maintenance/prediction workflow connected to operational decisions.
- `predictive-maintenance-reinforcement-learning` — maintenance decisions learned as a sequential control policy.
- `industrial-maintenance-markov-decision-process-python` — maintenance as an explicit MDP.

## Industrial monitoring bridge

- `industry-4.0-lab` — broader industrial-data portfolio containing condition-monitoring, predictive-maintenance, process-monitoring, and quality projects. It remains a separate curated lab rather than a maintenance-only repository.
- `time-series-intelligence-merlion` — time-series/anomaly tooling relevant to condition monitoring but not itself a maintenance optimizer.

## Digital-twin and dynamic-control bridge

- `industrial-digital-twin-system-optimization-python` — optimization around a digital-twin system.
- `dynamic-manufacturing-digital-twin-rl` — dynamic decision/control policy in a manufacturing digital-twin setting.
- `event-driven-continuous-reoptimization` — generic architecture for reoptimization after state changes; applicable to maintenance rescheduling when failures or asset states change.

## Why these repositories stay separate

The asset-management pipeline contains multiple distinct research problems:

1. detect or predict degradation;
2. estimate uncertainty or failure risk;
3. decide whether and when to intervene;
4. schedule maintenance resources;
5. react to new failures or operating-state changes.

A repository focused on one stage should not be merged with another merely because both mention maintenance or reliability.
