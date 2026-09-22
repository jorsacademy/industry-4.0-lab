# CP-SAT Horizon and Solver-Budget Sensitivity

## Purpose

The rolling-horizon CP-SAT controller introduces two operational design choices that must not be treated as arbitrary constants:

- the number of released jobs included in each reoptimization horizon;
- the maximum wall-clock solve budget allowed at each decision epoch.

A larger horizon may improve schedule quality by considering more interactions, but it increases model size. A larger solve budget may improve search quality, but it directly increases online decision latency. The sensitivity harness measures this quality-versus-compute trade-off on common random seeds.

## Default grid

The default nominal grid is:

- horizons: 4, 8, 12 released jobs;
- solve budgets: 20 ms, 50 ms, 100 ms per decision.

The default reference configuration is horizon 12 with a 100 ms solver budget.

Every grid point is evaluated on exactly the same stochastic seeds. OR-Tools uses one search worker and a fixed solver random seed, consistent with the main CP-SAT benchmark.

## Seed regime

Sensitivity analysis is a model-selection activity, so it uses the reserved **validation** range rather than final-test seeds. The repository convention is:

- training: `0-9999`;
- validation/model selection: `10000-10049`;
- nominal final test: `20000-20099`;
- distribution-shift final test: `30000-30099` per scenario family.

The sensitivity harness therefore defaults to `10000+`. Final nominal seeds must remain untouched until the CP-SAT operating point is frozen.

## Command

```bash
dmdtrl-or-sensitivity \
  --seeds 30 \
  --seed-start 10000 \
  --horizon 4 \
  --horizon 8 \
  --horizon 12 \
  --solver-seconds 0.02 \
  --solver-seconds 0.05 \
  --solver-seconds 0.10 \
  --reference-horizon 12 \
  --reference-solver-seconds 0.10 \
  --raw-output results/cpsat_sensitivity_runs.csv \
  --summary-output results/cpsat_sensitivity_summary.csv \
  --comparisons-output results/cpsat_sensitivity_comparisons.csv
```

## Outputs

The raw table retains one row per `(configuration, seed)` with the normal operational KPI schema plus:

- `cpsat_horizon`;
- `solver_budget_ms`.

The summary table adds bootstrap confidence intervals and a `pareto_optimal` flag. A configuration is Pareto-optimal when no other tested configuration has both:

- lower or equal mean priority-weighted tardiness; and
- lower or equal mean decision latency;

with at least one strict improvement.

The comparison table uses paired stochastic seeds to compare every non-reference configuration against the declared reference for:

- priority-weighted tardiness;
- mean online decision latency.

The same paired bootstrap, randomization-test, effect-size, and probability-of-superiority machinery used elsewhere in the repository is reused here.

## Interpretation

The sensitivity study is not a hyperparameter-tuning shortcut that selects the lowest observed WTT and then reports that same sample as final evidence. Its purposes are:

1. identify whether the OR controller is stable to reasonable horizon/budget choices;
2. quantify the marginal operational value of additional online compute;
3. identify non-dominated operating points for later PPO-vs-CP-SAT comparisons;
4. predeclare a practical CP-SAT configuration before the large final stress campaign.

A configuration should be selected using the validation seed set and then frozen before final evaluation seeds are analyzed.

## Next step

After the sensitivity grid is validated, one CP-SAT operating point will be frozen for the full comparative experiment. The next major scientific workload is then long-horizon PPO training across multiple training seeds followed by matched PPO-vs-CP-SAT-vs-fixed-rule evaluation on disjoint nominal and stress seeds.
