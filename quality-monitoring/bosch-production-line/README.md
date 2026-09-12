# Bosch Production Line — Early Defect Risk & Smart Quality

An Industry 4.0 smart-quality project built around the Bosch Production Line Performance competition data. The source contains anonymized measurements and measurement times for more than one million manufactured parts as they move through multiple production lines and stations. The objective is to identify rare parts that will fail quality control while preserving process chronology and exposing how much warning is available before a part completes its route.

## Why this project belongs here

This is substantially stronger than a generic manufacturing classification exercise:

- measurements are tied to real production lines and stations;
- each part has a traceable process route through sparse station measurements;
- date features encode when measurements were taken;
- defects are extremely rare, so ordinary accuracy is misleading;
- missingness is structural because different parts visit or trigger different stations;
- a useful deployment question is **how early** a risky part can be identified, not only whether its final label can be reproduced after all measurements are known.

The competition data page reports 1,183,747 labeled training parts and 6,879 failures (`Response=1`, about 0.58%). Numeric and date data are separated into very wide CSV files. Feature names retain line/station provenance, for example `L3_S36_F3939`.

## What is different from typical Kaggle solutions

The project does not use a random row split and does not optimize leaderboard accuracy in isolation.

### 1. Chronological evaluation

Every part receives a process start time from its earliest observed date feature. The benchmark is divided into four chronological blocks:

1. model fitting;
2. model selection;
3. probability calibration + operating-threshold selection;
4. untouched future test block.

Equal timestamps stay on the same side of a boundary. This makes the reported test result closer to a future-production deployment than a shuffled split.

### 2. Station-aware sparse feature engineering

Instead of treating thousands of anonymous columns as an arbitrary matrix, the feature builder parses the Bosch naming convention and aggregates measurements by physical station. For each line/station it can create:

- station observed/not observed;
- station time relative to the part's route start;
- number of numeric measurements observed;
- mean and standard deviation of station measurements;
- line-level station counts;
- elapsed process duration at the observation snapshot and overall missingness.

This keeps the cyber-physical process structure visible while reducing the raw dimensionality.

### 3. Early-warning prefixes

Feature tables can be built at 25%, 50%, 75%, and 100% of each part's observed route duration for retrospective early-warning evaluation. At an early prefix, measurements from stations that occur later in that part's route are masked, and only the elapsed duration up to that snapshot is exposed to the model; the final route end time and duration are not features.

This produces an operational trade-off curve:

```text
process fraction observed -> PR-AUC / precision / recall / MCC
```

A model that is slightly better at 100% but gives no useful warning until final inspection can be less valuable than a model that identifies risk earlier.

### 4. Rare-event metrics and thresholding

With roughly 0.58% failures, raw accuracy is not a useful primary KPI. The pipeline reports:

- average precision / PR-AUC;
- ROC-AUC as a secondary ranking metric;
- precision, recall and F1 at the selected operating threshold;
- Matthews correlation coefficient (MCC);
- Brier score after probability calibration;
- confusion-matrix counts.

The default operating point maximizes recall subject to a minimum precision requirement on the calibration period. That requirement is configurable and must be chosen against inspection/rework economics in a real plant.

### 5. Probability calibration

Model selection is performed on a chronological selection block. A separate later calibration block fits a Platt calibrator and chooses the operating threshold. The final test block is not used for model, calibration, or threshold selection.

## Data access and licensing

The Bosch files are governed by Kaggle's competition-specific rules. The raw data are therefore **not redistributed in this repository**.

Obtain access with your own Kaggle account, accept the applicable competition terms, configure the Kaggle CLI, then run:

```bash
cd quality-monitoring/bosch-production-line
pip install -r requirements.txt
python scripts/download_data.py
python scripts/validate_data.py --full
```

See [`SOURCES.md`](SOURCES.md) and [`data/README.md`](data/README.md).

## Build the station feature table

Full-route features:

```bash
python -m src.build_features --config config.yaml --prefix 1.0
```

A smaller development run can cap the number of raw rows without changing the feature logic:

```bash
python -m src.build_features --config config.yaml --prefix 1.0 --max-rows 100000
```

Build early-warning tables:

```bash
python -m src.build_features --config config.yaml --prefix 0.25
python -m src.build_features --config config.yaml --prefix 0.50
python -m src.build_features --config config.yaml --prefix 0.75
python -m src.build_features --config config.yaml --prefix 1.00
```

Feature extraction is chunked and writes Zstandard-compressed Parquet, so the full wide CSV does not need to be held in memory at once.

## Train and evaluate

```bash
python -m src.train --config config.yaml
```

The baseline compares:

- class-balanced logistic regression;
- class-balanced histogram gradient boosting.

The winner is chosen by PR-AUC on the model-selection time block. The chosen model is calibrated on the next time block and evaluated once on the future test block.

Generated outputs:

```text
reports/model_comparison.csv
reports/metrics.json
reports/test_predictions.csv
reports/feature_coefficients.csv   # selected logistic model only
artifacts/bosch_quality_model.joblib
```

No metric is claimed in this repository until the competition data has actually been obtained and the pipeline has been executed.

## Early-warning benchmark

After building all four prefix tables:

```bash
python -m src.early_warning --config config.yaml
```

This writes `reports/early_warning_tradeoff.csv`, allowing direct comparison of defect detection quality as more of the manufacturing route becomes observable.

## Repository layout

```text
bosch-production-line/
├── README.md
├── SOURCES.md
├── config.yaml
├── requirements.txt
├── Makefile
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_exploration.ipynb
├── reports/
│   └── README.md
├── scripts/
│   ├── download_data.py
│   └── validate_data.py
├── src/
│   ├── __init__.py
│   ├── schema.py
│   ├── data.py
│   ├── features.py
│   ├── build_features.py
│   ├── modeling.py
│   ├── train.py
│   └── early_warning.py
└── tests/
    ├── conftest.py
    ├── test_schema.py
    ├── test_features.py
    └── test_modeling.py
```

## Production interpretation

A deployed version should connect the calibrated risk score to an explicit intervention: route-specific inspection, targeted retest, containment, rework prioritization, or temporary process investigation. The correct threshold depends on the cost of false alarms versus escaped defects and should not be selected from leaderboard performance alone.

The next methodological extension would be to compare the station-aggregate baseline with a sparse raw-feature model and a route-aware sequence model under exactly the same chronological partitions and early-warning prefixes. Any more complex architecture should earn its added latency and maintenance cost with a reproducible out-of-time improvement.
