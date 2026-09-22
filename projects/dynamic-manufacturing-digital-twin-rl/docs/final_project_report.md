# Final Project Report — Dynamic Manufacturing Decision Twin

**Status: COMPLETE**  
**Completion date:** 2026-08-29

## 1. Executive conclusion

This repository is a completed industrial engineering / operations research decision-intelligence project built around manufacturing scheduling, simulation, mathematical optimization, reinforcement learning, and controlled statistical validation.

The project deliberately does not force an "AI wins" narrative. Two reinforcement-learning formulations were implemented, trained with multiple independent seeds, evaluated on common out-of-sample instances, and rejected at validation because they did not outperform strong fixed or OR baselines. The final FJSP benchmark therefore contains only controllers that had already passed their selection gates before final-test access.

On the untouched Phase-5 final block of **100 FJSP instances (`42000–42099`)**, the frozen rolling-horizon **CP-SAT H=4 / 100 ms** controller achieved the lowest mean priority-weighted tardiness:

- CP-SAT: **23.4672 WTT**, 95% bootstrap CI **[19.9513, 27.1294]**;
- Minimum Slack: **30.4134 WTT**, CI **[25.4342, 35.7992]**;
- Earliest Due Date: **32.4370 WTT**, CI **[27.0884, 38.4064]**.

Against the strongest fixed heuristic, Minimum Slack, CP-SAT reduced mean WTT by **6.9462 units / 22.84%**, paired 95% CI **[2.7670, 11.4262]**, paired randomization **p = 0.0008**, with **zero fallback** across the final campaign. Its cost is online computation: approximately **23.49 ms per decision** versus **0.014 ms** for Minimum Slack.

The completed decision result is therefore not "use RL." It is:

> Use rolling-horizon CP-SAT when roughly 20–25 ms of online compute is acceptable and priority-weighted tardiness is the primary objective; use Minimum Slack when near-zero decision latency is operationally dominant. The tested RL controllers do not justify additional complexity in this formulation.

## 2. Problem evolution

The repository contains two successive manufacturing scheduling research layers.

### Phase 1–4: dynamic heterogeneous parallel-machine scheduling

The original decision twin modeled stochastic arrivals, heterogeneous machines, due dates, priorities, sequence-related setup pressure, breakdown/repair delays, and quality-risk terms. It established the project methodology:

- event-driven simulation;
- deterministic dispatching baselines;
- rolling-horizon CP-SAT;
- PPO hyper-heuristic control;
- common random numbers;
- multi-training-seed RL evaluation;
- validation/final-test separation;
- out-of-distribution stress tests;
- bootstrap intervals and paired inference;
- compute/reliability accounting.

The final Phase-4 comparative campaign found that the strongest fixed composite dispatch rule beat the five-training-seed PPO mean on WTT in all ten final scenarios. That negative result was frozen instead of being tuned away.

### Phase 5: true flexible job shop scheduling

The project then introduced a separate true-FJSP stack rather than mutating the locked Phase-4 environment. The Phase-5 formulation includes:

- jobs with ordered multi-operation routes;
- strict operation precedence;
- operation-specific alternative eligible machines;
- machine-dependent processing times;
- dynamic job release times;
- family-dependent sequence setup effects;
- explicit `(job, operation, machine)` assignment decisions;
- event-driven advancement to the next feasible decision epoch;
- weighted tardiness, makespan, flow time, waiting, setup, utilization, and online decision latency accounting.

This richer formulation became the closing scientific scope of the repository.

## 3. Controllers evaluated

### Fixed FJSP dispatch operators

Eight deterministic feasible operators were used as transparent baselines and as the action primitives for the hyper-heuristic experiment:

1. Earliest Due Date;
2. Shortest Processing;
3. Minimum Setup;
4. Highest Priority;
5. Minimum Slack;
6. Critical Ratio;
7. Same Family First;
8. Weighted Tardiness Risk.

All operators act through the same FJSP simulator and therefore share the same precedence, eligibility, release, and resource-feasibility semantics.

### Rolling-horizon CP-SAT

The OR controller replans at each decision epoch. The frozen Phase-5 operating point is:

- job horizon: **4**;
- solver budget: **100 ms**;
- one CP-SAT search worker;
- fixed solver random seed;
- execute first decision and replan.

The configuration was selected on a separate validation block before final access and stored in `configs/fjsp_cpsat_validation_freeze.json`.

### Direct-action Maskable PPO

The first RL formulation selected directly from the dynamic feasible `(job, operation, machine)` action set using action masking.

Validation design:

- five independent PPO training seeds: `701, 1701, 2701, 3701, 4701`;
- 150,000 training timesteps per seed;
- validation instances: `41100–41129`;
- inference unit: independent training seed.

Result: rejected. Aggregate mean validation WTT was **82.5118**, and all five training realizations underperformed CP-SAT, EDD, and SPT. The training-seed-level mean paired improvement versus CP-SAT was **-62.0959**, CI **[-63.4498, -60.3620]**. No direct-action PPO realization was promoted to final testing.

Frozen evidence: `configs/fjsp_direct_ppo_validation_freeze.json`.

### PPO operator-selection hyper-heuristic

The second RL formulation reduced the action space to the eight always-feasible dispatch operators. PPO selected an operator, and that operator deterministically mapped the current state to a concrete feasible FJSP assignment.

Validation design:

- five independent PPO training seeds: `901, 1901, 2901, 3901, 4901`;
- 150,000 timesteps per seed;
- validation instances: `41200–41229`;
- common instance fingerprints across PPO and all baselines;
- algorithm inference at the training-seed level, never by best-seed cherry-picking.

Aggregate PPO validation WTT was **38.4618**. The controller beat several weaker dispatch rules, but failed the strong-baseline gate:

- versus Weighted Tardiness Risk: improvement **-6.6371**, CI **[-9.9216, -3.5945]**;
- versus EDD: improvement **-6.1630**, CI **[-9.4475, -2.9990]**;
- versus frozen CP-SAT: improvement **-17.3337**, CI **[-20.6181, -14.1696]**.

The hyper-heuristic was therefore not promoted to the final FJSP test. Representative seed `1901` was retained only for demo/deployment continuity, not scientific claims.

Frozen evidence: `configs/fjsp_hh_validation_decision.json`.

## 4. Data-partition discipline

Phase-5 seed regimes were intentionally separated before final-test access:

- development: `40000–40999`;
- CP-SAT tuning/selection: `41000–41029`;
- direct-action PPO validation: `41100–41129`;
- operator-selection PPO validation: `41200–41229`;
- reserved but unused v2 validation: `41300–41329`;
- final FJSP block: `42000–42099`.

The final block was not used to tune PPO, dispatch operators, the CP-SAT horizon, or the solver budget. After the one-time final benchmark, it is marked consumed and is not available for future model selection.

The abandoned `operator_selection_v2` experiment was archived without merge or validation. It is not part of the completed project evidence.

## 5. Final FJSP benchmark

### Protocol

The final benchmark evaluated exactly nine already-frozen non-RL controllers on the same 100 generated FJSP instances. For every seed, all policies used the same canonical instance fingerprint.

The panel therefore contains **900 policy-instance episodes**.

Primary metric: priority-weighted tardiness. Supporting metrics include makespan, mean flow time, online decision latency, and solver fallback rate.

Statistical reporting used:

- 5,000 bootstrap resamples for mean WTT intervals;
- 10,000 paired randomization permutations;
- paired differences on identical final instances;
- paired standardized effect size;
- probability of superiority.

### Final ranking

| Rank | Controller | Mean WTT | 95% WTT CI | Mean decision latency |
| ---: | --- | ---: | ---: | ---: |
| 1 | Rolling-Horizon CP-SAT | **23.4672** | **[19.9513, 27.1294]** | 23.4933 ms |
| 2 | Minimum Slack | 30.4134 | [25.4342, 35.7992] | 0.0139 ms |
| 3 | Earliest Due Date | 32.4370 | [27.0884, 38.4064] | 0.0090 ms |
| 4 | Critical Ratio | 36.5716 | [31.6531, 41.8447] | 0.0143 ms |
| 5 | Weighted Tardiness Risk | 37.0807 | [31.0336, 43.4308] | 0.0137 ms |
| 6 | Highest Priority | 43.1420 | [38.6059, 47.8443] | 0.0100 ms |
| 7 | Shortest Processing | 57.3151 | [51.2552, 63.6600] | 0.0097 ms |
| 8 | Same Family First | 74.2901 | [66.9821, 81.8338] | 0.0106 ms |
| 9 | Minimum Setup | 75.1594 | [67.6542, 82.6470] | 0.0101 ms |

CP-SAT fallback rate was **0.0**.

### Paired comparison with the strongest heuristic

Minimum Slack is the strongest fixed policy in the final panel.

For each final instance, define improvement as:

`Minimum Slack WTT - CP-SAT WTT`.

Results:

- mean improvement: **6.9462 WTT**;
- relative improvement: **22.84%**;
- 95% paired interval: **[2.7670, 11.4262]**;
- paired randomization p-value: **0.0008**;
- paired effect size `dz`: **0.3177**;
- probability of superiority: **0.60**;
- instance pairs: **100**.

Against EDD, CP-SAT improves WTT by **8.9698 / 27.65%**, paired interval **[4.6633, 13.7265]**, `p ≈ 0.0002`.

CP-SAT also has a positive paired WTT interval against each of the other six frozen heuristics.

## 6. Operational interpretation

The final result exposes a practical IE/OR trade-off instead of a single context-free winner.

### When to prefer CP-SAT

Choose the frozen rolling-horizon CP-SAT policy when:

- priority-weighted tardiness is strategically important;
- a roughly 20–25 ms decision budget is acceptable;
- a centralized optimization controller is operationally feasible;
- reproducible solver behavior and zero observed fallback at this problem scale are sufficient.

### When to prefer Minimum Slack

Choose Minimum Slack when:

- decision latency must be essentially negligible;
- infrastructure simplicity matters more than the measured WTT gap;
- the application cannot tolerate dependence on an online mathematical solver.

The final data show that the heuristic is approximately three orders of magnitude faster online, but pays roughly 22.8% more mean weighted tardiness on the final sample.

### What to conclude about RL

The correct conclusion is not that RL is intrinsically inappropriate for manufacturing scheduling. It is narrower and evidence-based:

- direct Maskable PPO did not learn a competitive assignment policy under the tested formulation and budget;
- operator-selection PPO improved the action abstraction but still did not clear strong WTT baselines;
- therefore neither controller earned final-test access;
- final performance claims are not made from the best RL seed, training reward, or a favorable scenario.

This is a useful decision-intelligence result: adaptive learning should be introduced only when it provides measurable operational value over simpler OR/control alternatives.

## 7. Reproducibility and provenance

Key frozen sources:

- `configs/fjsp_cpsat_validation_freeze.json` — frozen H4/100 ms FJSP CP-SAT policy;
- `configs/fjsp_direct_ppo_validation_freeze.json` — rejected direct-action PPO validation evidence;
- `configs/fjsp_hh_validation_decision.json` — rejected hyper-heuristic PPO validation evidence;
- `configs/fjsp_final_baseline_test.json` — predeclared final-test contract;
- `configs/project_completion.json` — project-level completion manifest.

Final benchmark provenance:

- source PR: **#28**;
- workflow run: **33271196539**;
- artifact ID: **9720170969**;
- artifact SHA-256: `8cdef3fb72df4f88156043b1e2aa54daf3f0ad92493385dca58bf1e814a4abf9`;
- final-test implementation merge: `a1071cfcea3f2b3e1b206ef3eb53325258dced03`.

The artifact contains:

- `fjsp_final_runs.csv`;
- `fjsp_final_summary.csv`;
- `fjsp_final_cpsat_comparisons.csv`;
- `fjsp_final_manifest.json`.

The final workflow verifies the exact seed set, exact policy panel, exact row count, final seed regime, RL exclusion, and no-retuning manifest flags before uploading the artifact.

## 8. Software quality gates

The completed repository includes:

- Python 3.10 / 3.11 / 3.12 CI;
- Ruff linting;
- pytest with an 85% project coverage gate;
- real OR-Tools integration tests;
- CP-SAT smoke and sensitivity tests;
- real Stable-Baselines3 / Maskable PPO integration workflows;
- multi-seed training artifacts and SHA-256 provenance;
- explicit validation and final-test boundary tests;
- deterministic common-instance fingerprint checks.

## 9. Limitations

The final Phase-5 conclusions are scoped to the implemented synthetic FJSP decision twin.

Important limits include:

- no live MES/ERP/IoT factory feed;
- synthetic stochastic instance generation rather than calibrated plant data;
- no stochastic machine breakdown process in the final FJSP stack;
- bounded problem scale (12 jobs, 5 machines under the frozen validation/final design);
- fixed dispatch operator library;
- no ALNS/neighborhood-search benchmark in the closing FJSP panel;
- final RL exclusion follows the predeclared validation gate and therefore does not estimate RL performance on `42000–42099`;
- the final seed block is consumed and must not be reused for future model selection.

These are extension opportunities, not unfinished obligations of this repository.

## 10. Project status

The research and portfolio scope is complete.

Completed deliverables include:

- manufacturing digital-twin simulation;
- parallel-machine and true-FJSP scheduling models;
- deterministic heuristic baselines;
- rolling-horizon mathematical optimization;
- direct and hyper-heuristic RL experiments;
- multi-seed validation;
- independent final evaluation;
- statistical uncertainty and paired inference;
- online compute and solver reliability accounting;
- CI, tests, manifests, hashes, and experiment artifacts;
- honest retention of negative RL evidence.

Any future live-data integration, API/dashboard deployment, richer disruptions, ALNS/GNN policies, or additional adaptive-controller research is explicitly optional follow-on work and is not required to regard this project as finished.
