# FJSP hyper-heuristic PPO validation result

## Decision

`operator_selection_v1` is **not promoted** to the Phase-5 final benchmark.

The decision is based only on the predeclared validation block `41200-41229`. The final block `42000-42099` remains embargoed and was not used for training, model selection, architecture selection within v1, or this promotion decision.

## Validation design

Five independent PPO training seeds were retained as the algorithm-level inference unit: `901`, `1901`, `2901`, `3901`, and `4901`. Each model used the same frozen 150,000-timestep PPO configuration and was evaluated on the same 30 canonical FJSP validation instances. Every instance was fingerprinted so PPO, the eight fixed dispatch operators, and the frozen rolling-horizon CP-SAT controller used identical problem realizations.

The representative model is seed `1901`, selected by the predeclared median-role rule. It is retained only for demo/deployment continuity. It is not the scientific PPO result.

## Main result

Across the five independent training seeds, PPO validation weighted tardiness was:

- mean: `38.462`
- median: `39.003`
- standard deviation across training seeds: `4.305`
- range: `34.046` to `44.909`
- mean online decision latency: `0.217 ms`

The strongest relevant comparators were:

| Controller | Mean weighted tardiness | Mean decision latency |
| --- | ---: | ---: |
| Frozen CP-SAT H4 / 100 ms | 21.128 | 21.507 ms |
| Weighted Tardiness Risk | 31.825 | 0.017 ms |
| Earliest Due Date | 32.299 | 0.012 ms |
| PPO hyper-heuristic v1 | 38.462 | 0.217 ms |

Lower weighted tardiness is better.

At the independent training-seed level, mean paired PPO improvement was negative against all three strong comparators:

| Baseline | Mean PPO improvement | Bootstrap CI | PPO training-seed win fraction |
| --- | ---: | ---: | ---: |
| Weighted Tardiness Risk | -6.637 | [-9.922, -3.595] | 0.0 |
| Earliest Due Date | -6.163 | [-9.447, -2.999] | 0.0 |
| Frozen CP-SAT | -17.334 | [-20.618, -14.170] | 0.0 |

PPO did beat weaker rules such as shortest processing, minimum setup, same-family-first, and on average highest-priority. That is not sufficient for promotion because the research standard is performance against the strongest credible baselines, not selective wins against weaker rules.

## Interpretation

The v1 action abstraction solved the feasibility problem of the earlier direct-action controller: PPO always chooses one of eight feasible dispatch operators, online latency is low, training is reproducible, and the full multi-seed validation pipeline works. The validation result nevertheless shows that the learned selector does not extract enough context-dependent value to outperform the best fixed rule, and remains substantially behind the frozen CP-SAT controller on weighted tardiness.

This is a useful negative result rather than a reason to weaken the benchmark. The final-test embargo is therefore preserved.

## Next iteration boundary

Any `operator_selection_v2` work must be treated as a new architecture iteration. It may use development data for diagnostics and implementation, but `41200-41229` is now consumed validation data and must not be reused for future model selection. Before any v2 multi-seed validation training or evaluation, a new independent validation block must be predeclared in a separate protocol change. The Phase-5 final block `42000-42099` remains untouched.

The main technical hypothesis for v2 should be evaluated on development data first: generic global state may be insufficient for contextual operator choice. A stronger selector should expose operator-conditioned candidate-action features (for example slack, processing duration, setup cost, machine readiness, priority and critical-ratio information for the action proposed by each operator) while retaining the fixed feasible operator action space. This is an architecture hypothesis, not post-hoc hyperparameter tuning of v1.

## Provenance

- Validation PR: `#24`
- Validation workflow run: `33270060963`
- Validated branch head: `e729ea80c542a323fecb80116b0a3ea15639c056`
- PR validation merge SHA used by Actions: `e2ceea6fe910468a525dd8895ea04bfbc5116d4f`
- Merged implementation SHA: `cecdbbe274bbd8b8c0ae569c055f488914545cc9`
- Campaign artifact ID: `9719874395`
- Campaign artifact SHA-256: `d6ae34b06d5cb88b11026f3a54b35e949d6099b527d88d90ce1cf037e9588be8`

The machine-readable decision is stored in `configs/fjsp_hh_validation_decision.json`.
