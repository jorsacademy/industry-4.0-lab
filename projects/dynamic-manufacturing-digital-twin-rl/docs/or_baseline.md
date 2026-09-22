# Rolling-Horizon CP-SAT Baseline

## Purpose

The CP-SAT baseline provides a classical operations-research comparator for the adaptive PPO hyper-heuristic and the eight fixed dispatching rules.

The intent is not to give the optimizer privileged simulator information. The OR policy uses the same online information boundary as an operational scheduler.

## Information set

At each decision epoch the solver may observe:

- jobs that have already been released into the queue;
- current time;
- due dates, priorities, processing requirements, product families, and quality-risk attributes of released jobs;
- currently available machines, their speeds, and their last processed family;
- the configured sequence-dependent setup duration.

The solver does **not** observe:

- future unreleased job arrivals;
- future random breakdown realizations;
- future repair durations.

Breakdowns remain stochastic simulator events and are only observed after they occur.

## Receding-horizon mechanism

The policy selects at most `max_jobs` released jobs using a deterministic urgency ordering based on due date, priority, arrival time, and job ID.

A CP-SAT model then plans those jobs over the machines that are available at the current decision epoch. Only the first job-machine assignment from the plan is executed. The simulator advances and the model is rebuilt at the next decision epoch.

This is therefore a true **rolling/receding-horizon** controller rather than a one-shot offline schedule.

## CP-SAT formulation

For every job in the horizon, the model chooses exactly one currently available machine. Processing duration is machine dependent:

`processing_time / machine_speed`.

Per-machine circuit constraints create a sequence through the jobs assigned to that machine. The sequence includes:

- initial setup from the machine's current family state;
- family-to-family setup transitions between consecutive jobs;
- non-overlap through precedence constraints on active sequence arcs.

The primary objective is priority-weighted tardiness. Setup burden and makespan are lower-order tie-breaking terms so that weighted tardiness remains dominant.

Continuous simulator times are converted to integer CP-SAT units with a configurable scale (default: 100 units per simulator time unit).

## Compute budget and determinism

Default settings:

- horizon: 12 released jobs;
- solver wall-clock budget: 0.10 seconds per decision;
- one CP-SAT search worker;
- fixed solver random seed.

One search worker is used to reduce run-to-run variability. Actual online decision latency is measured and reported alongside operational KPIs.

## Benchmark command

Install the OR dependency:

```bash
pip install -e ".[or,dev]"
```

Run nominal paired evaluation on the reserved `20000+` seed range:

```bash
dmdtrl-or \
  --seeds 30 \
  --seed-start 20000 \
  --horizon 12 \
  --solver-seconds 0.10 \
  --raw-output results/or_runs.csv \
  --summary-output results/or_summary.csv \
  --comparisons-output results/cpsat_comparisons.csv
```

The output uses the same KPI schema, bootstrap confidence intervals, paired randomization tests, effect sizes, and common-random-number seeds as the learned-policy research harness.

## Current limitations

The first CP-SAT baseline intentionally plans only released jobs on currently available machines. It does not yet reserve work for machines that are still busy, use arrival forecasts, or jointly model stochastic breakdown scenarios.

These restrictions make the first benchmark conservative and prevent future-information leakage. Later Phase 3 experiments can add explicitly declared forecast-aware variants and reoptimization-frequency studies as separate baselines rather than silently changing the information set.

## Next Phase 3 step

After the nominal CP-SAT implementation is validated, the same OR controller will be inserted into the distribution-shift suite so that the repository can produce matched comparisons among:

- PPO;
- rolling-horizon CP-SAT;
- eight fixed dispatching rules.

The primary research question is which controller class is preferable under different combinations of uncertainty, disruption, and online compute budget.
