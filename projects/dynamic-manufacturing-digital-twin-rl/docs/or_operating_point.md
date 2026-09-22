# CP-SAT Operating-Point Selection and Freeze

## Purpose

The CP-SAT horizon and online solve budget are model-selection choices. They must be selected before nominal final-test seeds (`20000+`) or stress final-test seeds (`30000+`) are analyzed.

The repository therefore reserves `10000–19999` for validation/model selection and provides a deterministic selection rule through `dmdtrl-or-select`.

## Validation campaign

The `OR Validation` GitHub Actions workflow runs the full nominal sensitivity grid on 30 common-random-number seeds:

- validation seeds: `10000–10029`;
- horizons: 4, 8, 12 released jobs;
- solve budgets: 20, 50, 100 ms per decision;
- bootstrap replicates: 2,000;
- paired randomization permutations: 5,000.

The workflow retains raw seed-level runs, bootstrap summaries, paired comparisons against H=12 / 100 ms, and the selected operating-point JSON manifest.

## Online timeout behavior

A bounded online optimizer can exhaust its solve budget before finding a first feasible CP-SAT solution. `UNKNOWN` is therefore treated as an operational timeout, not as model infeasibility.

On `UNKNOWN` only, the controller executes a deterministic one-step feasible fallback assignment based on immediate weighted tardiness, tardiness, setup, completion time, due date, priority, machine ID, and job ID. `INFEASIBLE` and model errors remain hard failures.

Every episode reports:

- fallback decision count;
- total decision count;
- solver fallback rate;
- solver success rate.

This makes short compute budgets measurable rather than silently crashing or being mislabeled as successful CP-SAT operation.

## Selection rule

The selector first restricts attention to configurations that are Pareto-optimal in mean priority-weighted tardiness and measured mean online decision latency. It then requires the mean solver fallback rate to be at most the predeclared default of 1%.

Let `WTT_best` be the lowest mean weighted tardiness among those reliable Pareto points. With the default 2% quality tolerance, a configuration is quality-acceptable when:

`WTT <= WTT_best × 1.02`

Among acceptable reliable configurations, choose the one with the lowest measured mean decision latency. Remaining ties are broken by lower WTT, lower fallback rate, lower solver budget, smaller horizon, then policy identifier.

This prevents a very small compute budget that mostly behaves as a heuristic fallback from being selected and described as the CP-SAT operating point.

## Data-integrity checks

`dmdtrl-or-select` refuses to create a manifest unless:

- validation seed count is positive;
- every seed is at least 10000 and strictly below 20000;
- at least two sensitivity configurations are present;
- every configuration contains the complete declared validation seed set;
- required sensitivity and solver-reliability fields are present;
- operational numeric values are finite and valid;
- at least one Pareto-optimal configuration exists;
- at least one Pareto configuration satisfies the declared fallback-rate limit.

These checks prevent partial grid runs, final-test leakage, and solver-timeout masking from silently determining the OR configuration.

## Freeze manifest

The generated JSON records selected horizon and solver budget, validation WTT and latency, solver fallback rate, reliability limit, quality tolerance and threshold, Pareto candidate counts, validation seed range, source files, Git commit SHA when available, and the exact textual selection rule.

After review, the selected horizon/budget is frozen and reused unchanged for nominal final-test and stress-test experiments.

## Scientific boundary

The validation artifact is model-selection evidence, not final performance evidence. Final claims must come from disjoint `20000+` nominal and `30000+` stress seeds after both CP-SAT and PPO settings are frozen.
