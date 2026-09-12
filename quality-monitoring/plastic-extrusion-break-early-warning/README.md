# Plastic Extrusion — Film-Break Early Warning

A leakage-aware predictive-quality benchmark for a real plastic-film extrusion process. The source contains a large multivariate production time series with hundreds of numeric process signals describing temperatures, pressures, dimensions, machine state and operating settings. Film breakage is represented by zero final-product thickness in the published data.

The project asks a stricter question than contemporaneous defect classification:

> Can process telemetry available before a break provide useful early warning of an upcoming film-break event?

## Source

Dataset: `podsyp/find-a-defect-in-the-production-extrusion-line` on Kaggle.

Public dataset catalogues describe the source as a real plastic-film extrusion process with **226,536 rows** and about **470 columns / 469 numeric process features**. The line heats and stretches granular plastic into a three-layer film; the process data include temperatures, pressure-related variables, machine/material dimensions, working times and other machine settings. The most important defect is film breakage. Public secondary documentation of the dataset defines a break when final-product thickness is zero.

The source is distributed under **CC BY-NC-ND 4.0**. Raw files are therefore downloaded transiently for non-commercial reproducible analysis and are not redistributed by this repository. Only code and aggregate benchmark reports are committed.

## Evaluation contract

The benchmark is designed to avoid the easiest leakage modes in this dataset:

1. The target is an **upcoming** film-break event, not the contemporaneous zero-thickness row.
2. Direct target / final-thickness columns and obvious defect-result columns are excluded from predictors.
3. Feature screening and imputation are fit on the historical training block only.
4. Model and warning horizon are selected before opening the final future block.
5. Chronological blocks with purge gaps are used instead of random row splitting.
6. Event-level metrics are reported in addition to row-level ranking metrics so that long fault episodes do not inflate performance.

The exact source schema is validated by the benchmark workflow before training. If no reliable time column exists, source row order is treated only as the published process sequence and horizons are reported in rows rather than falsely converted to seconds.

## Models

Candidate models are deliberately compact and auditable:

- class-weighted logistic regression;
- Extra Trees with class weighting;
- HistGradientBoosting on a train-only compact panel when supported by the validated source schema.

Selection uses average precision (PR-AUC) because upcoming breaks are expected to be rare. ROC-AUC is reported but is not the primary criterion.

## Diagnostics

The final report includes:

- PR-AUC, ROC-AUC and Brier score;
- precision, recall, F1, MCC and balanced accuracy at a frozen calibration threshold;
- event-level recall and warning lead distribution;
- alarms per 1,000 evaluated rows;
- train-to-future feature drift diagnostics;
- compact sensor / process-variable panel size when feature screening is beneficial.

Feature importance is observational. It is not treated as evidence that changing a process variable would prevent a break.

## Reproduce

```bash
cd quality-monitoring/plastic-extrusion-break-early-warning
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
reports/source_summary.json
reports/model_selection.csv
reports/metrics.json
reports/drift_diagnostics.csv
reports/feature_importance.csv
artifacts/extrusion_break_model.joblib
```

## Interpretation boundary

This is observational production data. A useful early-warning model would support inspection or operator attention; it would not by itself establish causal process adjustments. Any final future block reported by this project is consumed once and then closed to further feature, horizon, model or threshold selection.
