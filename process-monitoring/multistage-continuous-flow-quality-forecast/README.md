# Multi-Stage Continuous Flow — Lag-Aware Quality Forecasting & Drift Monitoring

An Industry 4.0 process-monitoring project built on real 1 Hz measurements from a high-speed continuous manufacturing line. The source contains one production run with parallel first-stage machines, a combiner, two serial second-stage machines, ambient conditions, process variables, setpoints, and 15 measured output locations at each stage.

The project asks a deliberately operational question:

> Using only process information available now, how well can the 15 second-stage quality measurements be forecast a short time into the future?

The benchmark treats transport/process delay, temporal dependence, measurement-system dropouts, uncertainty, and drift as first-class issues rather than randomly shuffling rows from one continuous run.

## Why this project adds something new

The repository already contains batch metallurgy, machine-condition, machining-response, and discrete production-quality projects. This project adds a different regime: a **continuous multi-stage production line** with parallel and serial equipment, 1 Hz process telemetry, downstream quality measurements, and an explicit time-alignment problem.

The source contains only one production run spanning several hours. That makes a random train/test split scientifically weak: adjacent rows are highly dependent and would leak local process state across the boundary. This project therefore uses ordered blocks, purge gaps, and one final future segment.

## Source and redistribution boundary

Source dataset: `supergus/multistage-continuousflow-manufacturing-process` on Kaggle.

The Kaggle page describes the data as coming from an actual production line near Detroit, Michigan. Stage 1 has Machines 1–3 operating in parallel before a combiner; Stage 2 has Machines 4–5 in series. Fifteen output locations are measured after each stage.

The source page labels the data files as **© Original Authors** rather than providing a permissive data license. Raw source files are therefore **not redistributed** in this repository. `scripts/download_data.py` obtains the public source through KaggleHub and keeps it untracked. Only code and aggregate derived benchmark reports are committed.

## Leakage-safe forecasting contract

### 1. Forecast horizon selected before final test

Candidate horizons are evaluated on the historical selection block only. A horizon of `h` seconds maps process features at time `t` to Stage-2 output measurements at `t+h`.

The selected horizon is therefore an empirical forecasting horizon for this dataset, not a claimed physical residence time for material moving through the line.

### 2. No future process variables

For a prediction made at row `t`, features are taken only from row `t`:

- ambient conditions;
- raw-material descriptors;
- Machine 1–5 process variables;
- combiner process variables;
- Stage-1 actual output measurements and setpoints;
- current Stage-2 setpoints.

Current Stage-2 **actual** outputs are excluded from the learned model so that the task remains an upstream/process-based downstream-quality forecast rather than a trivial autoregressive copy.

The persistence baseline is intentionally stronger: it predicts the future Stage-2 actual values using the currently observed Stage-2 actual values. The learned process model must be compared with this operational baseline.

### 3. Measurement-system dropout rule

The source documentation notes that zero output setpoints are associated with measurement-system dropouts. A target row is excluded when any Stage-2 target setpoint is non-positive. This rule is fixed before model selection.

### 4. Chronological blocks with purge gaps

After horizon alignment and the fixed validity rule, samples are divided in time order into:

- fit block;
- model/horizon selection block;
- conformal calibration block;
- untouched future test block.

A purge interval is removed between blocks to reduce direct dependence across boundaries. No test row is used for horizon, model-family, feature, or uncertainty selection.

## Models and metrics

Candidate models:

- Ridge regression with train-only median imputation and scaling;
- Extra Trees multi-output regression.

Selection minimizes the mean target-normalized MAE, where each of the 15 targets is normalized by its fit-block interquartile range. This prevents large-scale output channels from dominating model selection.

The final report includes:

- mean normalized MAE across 15 outputs;
- persistence and setpoint baselines;
- raw MAE/RMSE/R² for each output channel;
- 90% split-conformal marginal intervals calibrated on the later calibration block;
- interval coverage and normalized interval radius per target;
- moving-block bootstrap confidence interval for model-vs-persistence error improvement;
- feature-distribution drift diagnostics between fit and final test blocks using KS statistics and robust median shift.

No causal process-adjustment claim is made from feature importance.

## Reproduce

```bash
cd process-monitoring/multistage-continuous-flow-quality-forecast
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
reports/model_selection.csv
reports/metrics.json
reports/per_target_metrics.csv
reports/interval_coverage.csv
reports/drift_diagnostics.csv
reports/feature_importance.csv
artifacts/continuous_flow_quality_model.joblib
```

The fitted model is uploaded as a workflow artifact rather than committed as a binary.

## Interpretation boundary

This dataset contains a single several-hour production run. A chronological future holdout is much stronger than random-row validation, but it is still not evidence of generalization to a different day, product campaign, line, factory, controller revision, or maintenance state.

The selected forecast horizon is a data-driven alignment for the observed run, not a verified material-residence-time estimate. True residence-time identification would require line speed, equipment geometry, or traceable material markers.

The final future test is consumed once by the benchmark. Any later model redesign that uses its results requires a new independent production run for a valid improvement claim.
