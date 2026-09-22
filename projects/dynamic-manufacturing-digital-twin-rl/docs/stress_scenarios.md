# Distribution-Shift Stress Scenarios

The purpose of this suite is to measure controller robustness outside nominal training-like conditions. A learned policy is trained under nominal assumptions and then evaluated **without retraining** on controlled shifts. The rolling-horizon CP-SAT controller is reoptimized online in every scenario using only information available at the current decision epoch.

All controllers within a scenario use the same evaluation seeds. Scenario comparisons are therefore paired at the stochastic-seed level.

## Scenario definitions

| Scenario | Operational change |
| --- | --- |
| `nominal` | Base `EnvConfig` |
| `demand_120` | Arrival intensity ×1.20; mean interarrival divided by 1.20 |
| `demand_140` | Arrival intensity ×1.40; mean interarrival divided by 1.40 |
| `demand_160` | Arrival intensity ×1.60; mean interarrival divided by 1.60 |
| `breakdown_2x` | Breakdown probability ×2 |
| `breakdown_4x` | Breakdown probability ×4 |
| `tight_due_085` | Due-date allowance factors ×0.85 |
| `slow_machines_090` | Machine-speed range ×0.90 |
| `setup_2x` | Sequence-dependent setup time ×2 |
| `compound_stress` | Demand ×1.40, breakdown ×3, due-date allowance ×0.85, speed ×0.95, setup ×1.50 |

Breakdown probability is capped at 0.95 so an extreme custom multiplier cannot create an invalid probability.

## Controllers in the stress matrix

The stress harness supports three controller classes:

1. the eight deterministic dispatching rules;
2. an optional trained PPO hyper-heuristic supplied through `--model`;
3. the rolling-horizon CP-SAT controller enabled with `--include-cpsat`.

CP-SAT preserves the information boundary defined in [`or_baseline.md`](or_baseline.md): it sees released jobs and currently available machines, but not future job arrivals, breakdown realizations, or repair durations. It replans from the scenario-adjusted `EnvConfig` at every decision epoch.

## Evaluation protocol

Recommended stress-test seeds begin at `30000` and must remain disjoint from training and nominal final-test seeds.

For each scenario and controller retain raw seed-level metrics. Report at minimum:

- priority-weighted tardiness;
- on-time completion rate;
- mean waiting time;
- setup time;
- makespan;
- utilization;
- decision latency.

The command below evaluates fixed rules and CP-SAT on the same stress seeds:

```bash
dmdtrl-stress \
  --include-cpsat \
  --cpsat-horizon 12 \
  --cpsat-solver-seconds 0.10 \
  --seeds 50 \
  --seed-start 30000 \
  --raw-output results/stress_runs.csv \
  --summary-output results/stress_summary.csv \
  --cpsat-comparisons-output results/stress_cpsat_comparisons.csv
```

For a matched three-way study, supply the trained PPO model as well:

```bash
dmdtrl-stress \
  --model models/ppo_dispatcher.zip \
  --include-cpsat \
  --seeds 100 \
  --seed-start 30000 \
  --comparisons-output results/stress_ppo_comparisons.csv \
  --cpsat-comparisons-output results/stress_cpsat_comparisons.csv
```

When both optional controllers are present, PPO is compared against CP-SAT and every fixed rule. CP-SAT is independently compared against every fixed rule. Comparisons remain scenario-local and seed-paired.

## Interpretation

The scenarios are controlled interventions, not claims about one specific factory. They support questions such as:

- At what demand intensity does a nominally strong dispatching rule begin to fail?
- Does PPO preserve on-time performance as disruption frequency rises?
- Does CP-SAT's additional online compute produce enough operational improvement to justify its latency?
- Is a controller's nominal advantage explained only by setup reduction?
- Under compound stress, does the ranking between PPO, CP-SAT, and fixed rules change materially?

A single winner across all regimes is neither expected nor required. The useful scientific result is the operating region in which each controller class is preferable.

## Robustness curves

The central visualization should be a response curve rather than a single leaderboard, for example:

`arrival intensity -> paired improvement over the best fixed-rule baseline`

Equivalent curves should be produced for breakdown risk, due-date tightness, and other controlled stress dimensions. PPO and CP-SAT should also be compared directly while reporting their decision-time difference.

## Limitations

The current scenario system modifies parameters already represented by the parallel-machine simulator. It does not yet model:

- correlated machine failures;
- explicit urgent-order burst processes;
- supplier/material shortages;
- operator absenteeism;
- time-varying energy prices;
- multi-operation job routing.

Those effects should be added only with corresponding simulator logic and validation tests; they should not be represented by arbitrary reward penalties.
