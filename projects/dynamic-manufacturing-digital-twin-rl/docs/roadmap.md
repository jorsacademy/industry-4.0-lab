# Research Roadmap — Closed

**Project status: COMPLETE.**

This roadmap records the work that was completed in this repository. There are no remaining required research phases. Any service/dashboard layer, live-data integration, richer disruption model, ALNS/GNN controller, or further RL iteration is optional follow-on work and is not part of the completed project scope.

## Phase 1 — Validated manufacturing simulator and deterministic baselines

Status: **complete.**

Delivered:

- event-driven dynamic manufacturing simulator;
- stochastic arrivals, heterogeneous machines, setup effects, failures/repairs, priorities, due dates, and quality-risk terms in the original parallel-machine stack;
- deterministic dispatching baselines;
- operational KPI definitions;
- common-random-number evaluation;
- unit/integration tests and multi-version CI.

## Phase 2 — Rolling-horizon OR baseline

Status: **complete and frozen.**

Delivered:

- released-job-only rolling-horizon CP-SAT;
- job/machine assignment through the same simulator transition logic used by adaptive controllers;
- machine-dependent processing and setup treatment;
- online solve-budget and decision-latency accounting;
- deterministic timeout/fallback handling;
- solver reliability reporting;
- horizon × solver-budget sensitivity analysis;
- validation-based operating-point selection.

The original parallel-machine operating point is frozen in `configs/cpsat_operating_point.json`.

## Phase 3 — Parallel-machine PPO hyper-heuristic validation

Status: **complete and frozen.**

Delivered:

- Stable-Baselines3 PPO over eight dispatch rules;
- five independent long-horizon training runs;
- model/manifests hashes;
- common validation instances;
- bootstrap intervals, paired tests, effect sizes, probability of superiority, and training-seed dispersion;
- representative-model handling without best-seed substitution.

## Phase 4 — Parallel-machine nominal + distribution-shift final campaign

Status: **complete; final evidence frozen.**

Final result:

- `WEIGHTED_COMPOSITE` had lower mean priority-weighted tardiness than the five-training-seed PPO mean in all 10 final scenarios;
- PPO beat frozen CP-SAT robustly only in compound stress, but Weighted Composite still beat PPO in that scenario;
- the result therefore did not justify PPO complexity over the strongest fixed baseline;
- unfavorable training seeds and policy-collapse diagnostics were retained rather than hidden.

Evidence:

- `docs/final_comparative_results.md`;
- `results/final_comparative/`.

## Phase 5 — True flexible job-shop decision twin

Status: **complete; final evidence frozen.**

### FJSP core

Delivered:

- ordered multi-operation jobs;
- strict operation precedence;
- operation-specific alternative eligible machines;
- machine-dependent processing times;
- dynamic job release times;
- explicit `(job, operation, machine)` actions;
- event-driven advancement to the next feasible decision epoch;
- family/setup effects;
- reproducible stochastic FJSP generation;
- operation-level schedules and job-level weighted-tardiness/flow-time metrics.

### FJSP Gymnasium / action feasibility

Delivered:

- fixed-capacity action indexing;
- exact Boolean dynamic feasibility masks;
- normalized global/job/routing/machine state features;
- infeasible-action rejection instead of silent repair;
- decision traces;
- waiting/setup/final-tardiness reward components;
- simulator/action-mask equivalence tests.

### Strong FJSP OR comparator

Delivered:

- rolling-horizon FJSP CP-SAT;
- validation-only horizon/solve-budget selection on `41000–41029`;
- frozen final operating point `FJSP_CPSAT_H4_B100MS`;
- decision latency and fallback accounting.

### Direct-action Maskable PPO

Status: **implemented, validated, rejected.**

- training seeds: `701, 1701, 2701, 3701, 4701`;
- 150,000 timesteps per seed;
- validation seeds: `41100–41129`;
- aggregate validation WTT: `82.5118`;
- paired training-seed mean improvement versus frozen CP-SAT: `-62.0959`, CI `[-63.4498, -60.3620]`;
- no training realization passed the final-test gate.

Frozen evidence: `configs/fjsp_direct_ppo_validation_freeze.json`.

### PPO operator-selection hyper-heuristic

Status: **implemented, validated, rejected.**

- action space: eight deterministic feasible dispatch operators;
- training seeds: `901, 1901, 2901, 3901, 4901`;
- 150,000 timesteps per seed;
- validation seeds: `41200–41229`;
- aggregate validation WTT: `38.4618`;
- versus Weighted Tardiness Risk: `-6.6371`, CI `[-9.9216, -3.5945]`;
- versus frozen CP-SAT: `-17.3337`, CI `[-20.6181, -14.1696]`;
- not promoted to final testing.

Frozen evidence: `configs/fjsp_hh_validation_decision.json`.

### Operator-selection v2

Status: **archived without merge or validation.**

A new validation block (`41300–41329`) had been reserved, but the project was intentionally closed instead of starting another research iteration. PR #27 was closed without merge and the reserved block was never opened.

### One-time FJSP final benchmark

Status: **complete and consumed.**

Final seeds: `42000–42099`.

The final panel contained only the eight frozen dispatch rules plus the already-frozen FJSP CP-SAT controller. Both RL formulations were excluded because they failed validation before final access.

Final WTT ranking:

1. **Rolling-Horizon CP-SAT — 23.4672**;
2. **Minimum Slack — 30.4134**;
3. **Earliest Due Date — 32.4370**;
4. Critical Ratio — 36.5716;
5. Weighted Tardiness Risk — 37.0807;
6. Highest Priority — 43.1420;
7. Shortest Processing — 57.3151;
8. Same Family First — 74.2901;
9. Minimum Setup — 75.1594.

Against Minimum Slack, CP-SAT reduced mean WTT by **22.84%**, paired CI **[2.7670, 11.4262]**, `p = 0.0008`, with **0.0 fallback**. Mean decision latency was **23.49 ms** for CP-SAT versus **0.014 ms** for Minimum Slack.

Final artifact provenance:

- PR #28;
- workflow run `33271196539`;
- artifact `9720170969`;
- SHA-256 `8cdef3fb72df4f88156043b1e2aa54daf3f0ad92493385dca58bf1e814a4abf9`.

The final block is now consumed. No controller may be retuned using these outcomes.

## Completion decision

The repository has answered its core decision question with a defensible operational recommendation:

- use rolling-horizon CP-SAT when priority-weighted tardiness justifies a roughly 20–25 ms decision budget;
- use Minimum Slack as the strongest near-zero-latency fixed alternative;
- do not promote either tested RL formulation for this FJSP problem under the evaluated training/validation design.

The project is therefore complete rather than "waiting for RL to win."

Completion manifest: `configs/project_completion.json`.

Final report: `docs/final_project_report.md`.

## Optional work outside the completed scope

The following are possible future projects or extensions, not open tasks in this repository:

- MES/ERP/IoT live-state integration;
- calibrated real-factory datasets;
- stochastic FJSP machine breakdowns and repair decisions;
- ALNS/neighborhood-search FJSP comparators;
- graph/attention RL policies on larger variable-size instances;
- FastAPI/dashboard decision-twin service layer;
- joint production/maintenance decisions;
- dynamic EV routing/charging;
- multi-echelon supply-chain decision twins.

The reusable portfolio methodology remains:

**mathematical/OR baseline + simulator/digital twin + adaptive policy when justified + compute/reliability accounting + paired out-of-sample validation.**
