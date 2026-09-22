# Research Protocol

This repository treats reinforcement learning as one decision method among several Operations Research alternatives. Claims are based on paired out-of-sample experiments, not training reward.

## Primary research question

Under what levels of demand variability and operational disruption does an adaptive scheduling hyper-heuristic provide material operational value over fixed dispatching rules and rolling-horizon optimization?

## Experimental unit

One stochastic seed defines one complete manufacturing scenario. Every policy in a comparison is evaluated on the same seed. This common-random-number design reduces noise in pairwise comparisons.

## Data splits

Training seeds, nominal test seeds, and distribution-shift test seeds must be disjoint. Model selection must not use final test seeds.

Recommended convention:

- training: seeds 0-9999 sampled during learning;
- validation: fixed seeds 10000-10049;
- nominal test: fixed seeds 20000-20099;
- distribution-shift test: fixed seeds 30000-30099 per scenario family.

The exact counts may change with compute budget, but the separation must remain explicit.

## Required baselines

RL should not be compared only with FIFO. The minimum deterministic baseline set is:

- FIFO;
- highest priority first;
- earliest due date;
- shortest processing time;
- same-family first;
- minimum setup;
- critical ratio;
- weighted composite dispatching.

The next research phase adds a rolling-horizon CP-SAT or MILP baseline.

## Primary KPIs

Primary decision-quality metrics:

- priority-weighted tardiness;
- on-time completion rate;
- mean waiting time.

Secondary operational metrics:

- makespan;
- setup time;
- utilization;
- disruption/repair time;
- decision latency.

Reward is a training mechanism and is not a primary business KPI.

## Statistical reporting

For every policy and KPI report:

- number of seeds;
- mean;
- sample standard deviation;
- bootstrap 95% confidence interval.

For candidate-vs-baseline comparisons use paired seeds and report:

- mean paired improvement, oriented so positive favors the candidate;
- bootstrap 95% CI of paired improvement;
- two-sided paired randomization/permutation p-value;
- paired standardized effect size (Cohen's dz);
- probability of superiority;
- percent improvement relative to the baseline mean.

Statistical significance alone is insufficient. Operational effect size and decision latency must also be reported.

## Distribution-shift matrix

A trained policy should be evaluated under at least the following shifts:

1. arrival intensity: nominal, +20%, +40%, +60%;
2. breakdown probability: nominal, moderate, severe;
3. due-date tightness: nominal and tighter windows;
4. machine-speed profiles unseen during training;
5. product-family mixes with higher setup pressure;
6. urgent-order bursts.

The central scientific output is the relationship between uncertainty/disruption level and the relative advantage of adaptive policies.

## Reproducibility requirements

Every experiment should record:

- code commit SHA;
- environment configuration;
- model/training seed;
- evaluation seed;
- trained-model identifier;
- library versions;
- wall-clock decision latency;
- raw seed-level KPIs.

Aggregated tables are never a substitute for raw paired results.

## Anti-hype acceptance criteria

An RL result should be described as useful only if it satisfies all of the following:

1. beats at least one strong non-RL baseline on the predeclared primary KPI;
2. the paired confidence interval supports a non-trivial improvement;
3. feasibility is preserved;
4. online decision time is operationally acceptable;
5. performance remains credible under at least one meaningful distribution shift;
6. training and inference costs are disclosed.

If these conditions are not met, the correct conclusion may be that a classical OR policy is preferable.
