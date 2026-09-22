# Dynamic Manufacturing Decision Twin — OR, RL & Flexible Job Shop Scheduling

**Status: COMPLETE — portfolio-ready research project.**

A research-grade Industrial Engineering / Operations Research project that combines an event-driven manufacturing digital twin, flexible job-shop scheduling, rolling-horizon mathematical optimization, reinforcement learning experiments, and controlled statistical validation.

The repository is intentionally evidence-driven rather than "AI-first": two RL formulations were implemented and rejected at independent validation gates, while a frozen rolling-horizon CP-SAT controller won the final true-FJSP benchmark.

## Final result

The closing Phase-5 benchmark used **100 previously untouched FJSP instances (`42000–42099`)** and exactly nine frozen non-RL controllers. All controllers saw the same canonical instance for each seed. No controller was retuned after final-test access.

| Rank | Controller | Mean weighted tardiness | 95% bootstrap CI | Mean decision latency |
| ---: | --- | ---: | ---: | ---: |
| 1 | **Rolling-Horizon CP-SAT** | **23.4672** | **[19.9513, 27.1294]** | 23.4933 ms |
| 2 | Minimum Slack | 30.4134 | [25.4342, 35.7992] | 0.0139 ms |
| 3 | Earliest Due Date | 32.4370 | [27.0884, 38.4064] | 0.0090 ms |
| 4 | Critical Ratio | 36.5716 | [31.6531, 41.8447] | 0.0143 ms |
| 5 | Weighted Tardiness Risk | 37.0807 | [31.0336, 43.4308] | 0.0137 ms |
| 6 | Highest Priority | 43.1420 | [38.6059, 47.8443] | 0.0100 ms |
| 7 | Shortest Processing | 57.3151 | [51.2552, 63.6600] | 0.0097 ms |
| 8 | Same Family First | 74.2901 | [66.9821, 81.8338] | 0.0106 ms |
| 9 | Minimum Setup | 75.1594 | [67.6542, 82.6470] | 0.0101 ms |

Against the strongest fixed heuristic, **Minimum Slack**, frozen CP-SAT reduced mean weighted tardiness by **22.84%** (`6.9462` WTT units), with paired 95% CI **[2.7670, 11.4262]**, paired randomization **p = 0.0008**, `n = 100`, and **0.0 fallback rate**.

The trade-off is computational: CP-SAT uses about **23.49 ms per online decision**, while Minimum Slack uses about **0.014 ms**. The operational recommendation is therefore conditional rather than universal:

- use CP-SAT when weighted tardiness is the dominant KPI and a ~20–25 ms decision budget is acceptable;
- use Minimum Slack when effectively instantaneous decisions and minimal infrastructure are more important than the observed WTT gap.

Full scientific narrative: [`docs/final_project_report.md`](docs/final_project_report.md).

Machine-readable completion evidence: [`configs/project_completion.json`](configs/project_completion.json).

## What was built

### 1. Event-driven manufacturing simulation / digital twin

The project began with dynamic heterogeneous parallel-machine scheduling and then added a separate true-FJSP stack. Across the repository the simulator/evaluation infrastructure covers:

- stochastic/dynamic job release;
- heterogeneous machines;
- priorities and due dates;
- sequence/family setup effects;
- event-driven decision epochs;
- multi-operation jobs with strict precedence;
- operation-specific alternative eligible machines;
- machine-dependent processing times;
- explicit `(job, operation, machine)` scheduling actions;
- weighted tardiness, makespan, flow-time, waiting, setup, utilization, and decision-latency KPIs.

### 2. Strong deterministic scheduling baselines

The true-FJSP closing panel contains eight feasible dispatch operators:

- Earliest Due Date;
- Shortest Processing;
- Minimum Setup;
- Highest Priority;
- Minimum Slack;
- Critical Ratio;
- Same Family First;
- Weighted Tardiness Risk.

These provide transparent, extremely fast baselines instead of comparing learning only against weak FIFO-style rules.

### 3. Rolling-horizon CP-SAT

The FJSP optimizer replans at each decision epoch and executes only the first decision before replanning. The final configuration was frozen before final access:

- horizon: **4 jobs**;
- solve budget: **100 ms**;
- single search worker;
- fixed solver seed;
- final fallback rate: **0.0**.

Frozen selection provenance: [`configs/fjsp_cpsat_validation_freeze.json`](configs/fjsp_cpsat_validation_freeze.json).

### 4. Reinforcement learning — tested, not promoted

Two RL formulations were implemented with real multi-seed training and reproducibility manifests.

#### Direct-action Maskable PPO

Maskable PPO selected directly from the dynamic feasible FJSP assignment set.

- training seeds: `701, 1701, 2701, 3701, 4701`;
- 150,000 timesteps per seed;
- independent validation instances: `41100–41129`;
- aggregate WTT across training seeds: **82.5118**;
- mean paired improvement versus frozen CP-SAT: **-62.0959**, CI **[-63.4498, -60.3620]**.

It was rejected before final testing. See [`configs/fjsp_direct_ppo_validation_freeze.json`](configs/fjsp_direct_ppo_validation_freeze.json).

#### PPO operator-selection hyper-heuristic

PPO selected one of the eight always-feasible dispatch operators, which then produced the concrete FJSP assignment.

- training seeds: `901, 1901, 2901, 3901, 4901`;
- 150,000 timesteps per seed;
- independent validation instances: `41200–41229`;
- aggregate WTT across training seeds: **38.4618**;
- versus Weighted Tardiness Risk: **-6.6371**, CI **[-9.9216, -3.5945]**;
- versus frozen CP-SAT: **-17.3337**, CI **[-20.6181, -14.1696]**.

It improved on several weaker operators but did not clear the strong-baseline gate, so it was also excluded from final testing. See [`configs/fjsp_hh_validation_decision.json`](configs/fjsp_hh_validation_decision.json).

No best RL training seed is substituted for the algorithm-level result.

## Scientific design

The project uses explicit data partitions so that model selection and final reporting are separated.

Phase-5 seed regime:

- development: `40000–40999`;
- CP-SAT tuning/selection: `41000–41029`;
- direct-action PPO validation: `41100–41129`;
- operator-selection PPO validation: `41200–41229`;
- reserved but unused v2 validation: `41300–41329`;
- final FJSP test: `42000–42099` — **consumed and closed**.

The final FJSP block was opened once after controller selection was complete. It must not be used for future tuning or RL model selection.

Core evaluation practices include:

- common random numbers / common instance fingerprints;
- multiple independent RL training seeds;
- fixed validation and final-test boundaries;
- bootstrap confidence intervals;
- paired randomization inference;
- effect sizes and probability of superiority;
- online decision latency;
- solver fallback/reliability accounting;
- model, manifest, and artifact SHA-256 provenance;
- CI-enforced experiment contracts.

## Project evolution

The repository preserves two complete research stages.

### Phase 1–4 — dynamic heterogeneous parallel machines

The original simulator included stochastic arrivals, setups, failures/repairs and OOD stress scenarios. Five-seed PPO was compared with eight fixed rules and rolling-horizon CP-SAT over nominal and distribution-shift final campaigns.

Result: the strongest fixed composite rule beat the PPO training-seed mean on weighted tardiness in all ten final scenarios. This negative RL result was frozen rather than tuned away.

Evidence: [`docs/final_comparative_results.md`](docs/final_comparative_results.md) and [`results/final_comparative/`](results/final_comparative/).

### Phase 5 — true FJSP

The richer FJSP stack introduced precedence, alternative machines, multi-operation routing and dynamic release times. It produced the project’s closing decision result: **rolling-horizon CP-SAT is the best WTT controller in the final frozen panel; Minimum Slack is the low-latency alternative; both tested RL architectures fail their validation gates.**

## Reproducibility

### Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[rl,or,dev,analysis]"
```

### Quality gates

```bash
ruff check src tests
pytest --cov=dmdtrl --cov-report=term-missing --cov-fail-under=85
```

CI runs these checks on Python 3.10, 3.11 and 3.12. Python 3.11 also executes real OR-Tools research smoke tests.

### Audit the frozen FJSP final benchmark

The final workflow is retained for **audit/reproduction only**. The seed block is already consumed and must not be used to select or retune controllers.

```bash
python -m dmdtrl.fjsp_final \
  --config configs/fjsp_final_baseline_test.json \
  --environment-design configs/fjsp_hh_validation_design.json \
  --cpsat-freeze configs/fjsp_cpsat_validation_freeze.json \
  --rl-decision configs/fjsp_hh_validation_decision.json \
  --output-root results/fjsp_final_audit \
  --bootstrap 5000 \
  --permutations 10000
```

Final benchmark provenance:

- PR: **#28**;
- workflow run: **33271196539**;
- artifact ID: **9720170969**;
- artifact SHA-256: `8cdef3fb72df4f88156043b1e2aa54daf3f0ad92493385dca58bf1e814a4abf9`;
- benchmark implementation merge: `a1071cfcea3f2b3e1b206ef3eb53325258dced03`.

## Repository map

Key modules and evidence:

- `src/dmdtrl/fjsp_simulator.py` — deterministic event-driven FJSP transition core;
- `src/dmdtrl/fjsp_env.py` — masked Gymnasium FJSP environment;
- `src/dmdtrl/fjsp_optimization.py` — rolling-horizon CP-SAT controller;
- `src/dmdtrl/fjsp_operators.py` — eight deterministic FJSP dispatch operators;
- `src/dmdtrl/fjsp_hyperheuristic_env.py` — operator-selection RL environment;
- `src/dmdtrl/fjsp_final.py` — frozen one-time final benchmark/audit runner;
- `configs/` — frozen data boundaries, model-selection decisions and provenance;
- `docs/final_project_report.md` — final scientific report;
- `configs/project_completion.json` — machine-readable project completion state;
- `.github/workflows/` — CI, RL/OR campaigns and final benchmark gates.

## What this project demonstrates

From an Industrial Engineering / OR perspective, the repository demonstrates the full decision-model lifecycle rather than a single optimization notebook:

1. formalize operations and constraints;
2. implement an event-driven digital twin;
3. construct strong interpretable baselines;
4. build a rolling-horizon mathematical optimizer;
5. add adaptive RL controllers where sequential adaptation is plausible;
6. evaluate all controllers on common out-of-sample instances;
7. account for compute latency and reliability;
8. reject models that fail independent validation;
9. open a blinded final test once;
10. freeze the resulting operational recommendation with provenance.

The most important portfolio signal is methodological: **model complexity must earn its place against strong OR and heuristic alternatives.**

## Limitations

The final Phase-5 result is scoped to the synthetic FJSP decision twin in this repository. It is not a claim about all manufacturing plants or all FJSP distributions.

Current limitations include:

- no live MES/ERP/IoT integration;
- no plant-calibrated real production data;
- no stochastic breakdown process in the closing FJSP stack;
- bounded frozen problem scale (12 jobs, 5 machines);
- no ALNS/neighborhood-search comparator in the final FJSP panel;
- no RL result on final seeds because both RL architectures were correctly rejected at validation.

These are optional follow-on extensions, not unfinished requirements of this completed repository.

## Project status

All required research, validation, final-test and documentation gates for this repository are complete. The experimental `operator_selection_v2` branch was archived without merge and without opening its reserved validation block.

Future dashboards, live-data connectors, richer disruptions, ALNS/GNN policies or additional RL research should be treated as separate extensions or new repositories rather than prerequisites for completion.

See [`docs/roadmap.md`](docs/roadmap.md) for the closed research roadmap.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
