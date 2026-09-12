# SECOM Semiconductor Yield — Process Monitoring & Sensor-Panel Rationalization

An Industry 4.0 smart-quality project built on the UCI SECOM semiconductor manufacturing dataset. The source represents real production entities observed through hundreds of anonymous sensor/process-measurement variables, with pass/fail yield labels and a timestamp for each test point.

The project is intentionally not another shuffled pass/fail classifier. Its operational question is:

> How small can the monitored process-variable panel become while retaining useful rare-failure detection on future production?

## Dataset

Canonical source: UCI Machine Learning Repository, SECOM (DOI `10.24432/C54305`).

The raw release contains:

- 1,567 production entities;
- 590 numeric process/sensor columns in `secom.data` (the UCI page describes 591 attributes at the metadata level);
- 1,463 pass records (`-1`);
- 104 fail records (`1`);
- missing values of varying severity;
- one date/time stamp per production entity.

UCI licenses the dataset under **CC BY 4.0**. This repository downloads the canonical UCI archive at use time and records file hashes.

## Why it adds something new

Bosch Production Line Performance is a large station-route early-warning problem. SECOM is a different industrial problem: a small-`n`, high-dimensional process-monitoring system where many signals are redundant, noisy, constant, or heavily missing. The useful engineering output is therefore not only a defect probability but also a compact, auditable set of process variables worth retaining in the monitoring panel.

Because the process variables are anonymized, selected features are **diagnostic priorities**, not proven physical root causes.

## Experimental protocol

### 1. Strict chronological partitions

Records are ordered by the source timestamp and divided into four non-overlapping blocks:

1. fit;
2. model/panel selection;
3. threshold calibration;
4. final future test.

Equal timestamps are never split across blocks. No future block is used to fit imputation values, missingness filters, feature scores, model parameters, or the operating threshold.

### 2. Train-only data-quality filtering

On the fit block only, the pipeline removes variables above the configured missing-value rate, all-missing variables, and constant variables. Remaining values are median-imputed using fit-block medians.

### 3. Process-variable panel search

An ANOVA ranking is fitted on the fit block after train-only filtering/imputation. The project evaluates several panel sizes plus the full eligible set with class-balanced logistic regression and Extra Trees.

The primary selection metric is **average precision / PR-AUC**. Among configurations within a configurable absolute PR-AUC tolerance of the best selection result, the smallest process-variable panel is preferred. This makes monitoring burden an explicit model-selection consideration rather than an afterthought.

### 4. Rare-event operating point

After model/panel selection, the chosen configuration is refit on fit + selection history. The later calibration block sets a decision threshold that meets a configurable minimum recall while maximizing precision.

The final future block is evaluated exactly once with PR-AUC, ROC-AUC, precision, recall, F1, MCC, balanced accuracy / balanced error rate, Brier score, and confusion-matrix counts.

### 5. Process-drift diagnostic

For every selected variable, the project compares historical versus final-test missing rate, median, interquartile range, and robust median shift in historical-IQR units. This is a drift diagnostic, not causal root-cause analysis.

## Reproduce

```bash
cd quality-monitoring/secom-semiconductor-yield
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
reports/panel_model_comparison.csv
reports/metrics.json
reports/selected_process_variables.csv
reports/drift_report.csv
reports/test_predictions.csv
artifacts/secom_yield_model.joblib
```

`panel_model_comparison.csv` exposes the predictive-quality versus monitoring-panel-size trade-off rather than hiding feature count behind a single final model.

## Portfolio standard

No benchmark number is claimed until the canonical data has actually been downloaded and the declared chronological experiment has run. The benchmark workflow stores real results after successful execution.

A strong future extension would add cost information for individual measurement points, then replace the feature-count proxy with an explicit acquisition/monitoring-cost objective.
