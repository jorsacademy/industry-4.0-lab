# CNC Turning — Surface Integrity, Cutting Forces & Process Trade-offs

An Industry 4.0 machining project built on real CNC turning experiments with measured cutting forces, surface roughness, process setpoints and quantified flank-wear levels. The source was produced in a manufacturing laboratory on AISI H13 steel using a CNC turning center, a three-component dynamometer, a roughness tester and microscope-based tool-wear measurements.

This project is deliberately different from the repository's existing CNC tool-wear classifier. The existing project uses 10 Hz controller/electrical telemetry to infer worn vs. unworn condition. Here the problem is **process-quality decision support**: estimate surface roughness and cutting load from experimentally controlled settings, quantify the additional value of in-process force sensing, and expose Pareto-efficient operating points instead of optimizing a single metric.

## Source experiment

The dataset contains three CSV tables. The production benchmark uses the two planned experiments:

- `Exp1.csv`: 54 machining runs from a `3^3` full-factorial design with two replicas; factors are depth of cut (`ap`), cutting speed (`vc`) and feed (`f`).
- `Exp2.csv`: 48 runs with two depth-of-cut levels, four feed levels and three flank-wear levels (`TCond = 0.0, 0.1, 0.3 mm`), again with two replicas; cutting speed is fixed at 350 m/min.

Each machining run has six surface-roughness measurements taken at different positions. Force and process variables are repeated across those six rows. Treating those rows as independent train/test samples would create severe leakage.

## Methodology

### 1. Run-level aggregation

The six roughness positions are collapsed to one physical machining-run record. The pipeline retains:

- process setpoints: `ap`, `vc`, `f`, `TCond`;
- measured force components and resultant force;
- mean and standard deviation of `Ra` across the six positions;
- other available roughness summaries;
- experiment, tool and block metadata when present.

The modeling unit is therefore a machining run, not an individual roughness probe position.

### 2. Replicate-safe validation

Cross-validation groups runs by their physical process condition `(ap, vc, f, TCond)`. Replicate runs of the same condition are always assigned to the same fold. This asks a harder and more useful question than a random row split: can the surrogate generalize to process settings not seen during fitting?

### 3. Two roughness estimators

The benchmark separates two deployment modes:

- **setpoint-only surrogate**: predicts mean surface roughness from process settings before cutting;
- **force-assisted estimator**: adds measured force channels available during cutting.

The difference in cross-validated MAE quantifies whether force sensing provides useful information beyond the programmed machining parameters.

### 4. Cutting-load surrogate

A separate model predicts resultant cutting force from the process settings. This provides a load objective that can be evaluated before selecting a machining recipe.

### 5. Experiment-supported Pareto frontier

After model selection, the final surrogates score only process recipes that are present in the experimental design. The decision layer does **not** extrapolate to arbitrary continuous settings.

A recipe is Pareto efficient when no other experimentally supported recipe simultaneously gives:

- lower predicted surface roughness;
- lower predicted cutting force; and
- higher throughput proxy `ap × f × vc`.

Pareto fronts are reported by tool-wear regime so that a setting that is attractive with a new tool is not silently assumed to remain attractive near end of life.

## Reproduce

```bash
cd process-optimization/cnc-turning-quality-force
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/validate_data.py
python -m src.train --config config.yaml
```

Windows activation:

```bash
.venv\Scripts\activate
```

## Outputs

```text
reports/model_comparison.csv
reports/metrics.json
reports/oof_predictions.csv
reports/pareto_candidates.csv
reports/run_level_summary.csv
reports/feature_importance.csv
artifacts/cnc_turning_surrogates.joblib
```

The benchmark workflow downloads the source transiently, validates the published experiment counts, runs the grouped benchmark and commits only derived reports. Raw source files are not redistributed from this repository.

## Interpretation boundaries

This is a controlled experimental dataset, not a continuously streaming plant historian. The tool condition in `Exp2` is an experimental factor, not an automatically inferred wear state. The models therefore support process-response estimation and decision analysis under the tested regime; they do not establish universal tool-life laws.

The source variables are physically meaningful, but causal claims still require machining-domain validation. A Pareto-efficient recipe is a decision-support candidate, not an automatic production recommendation.
