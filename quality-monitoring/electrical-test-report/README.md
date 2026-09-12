# Electrical Test Report — Smart Quality Monitoring

An Industry 4.0 smart-quality project built around high-volume end-of-line electrical test records from electronic assembly. Each production record includes lot traceability, test time, a final pass/fail-style result, and 19 anonymized electrical measurements (`F1`–`F19`).

## Scope decision

This dataset is included. It is not continuous machine-condition telemetry like the CNC project, but it is still cyber-physical production data: an automated electrical test process is measuring manufactured units, attaching production-lot/time context, and generating a quality decision. That is directly usable for smart quality, zero-defect manufacturing, drift detection, lot-risk monitoring, and adaptive inspection.

The main limitation is semantic rather than structural: the `F1`–`F19` channels are anonymized, so the project can identify predictive electrical signatures but cannot claim a physical root cause or engineering unit for an individual channel without additional documentation.

## Verified dataset characteristics

A 2026 peer-reviewed study using the same Kaggle file reports:

- source file: `Electrical_Test_Report.csv`, UTF-16 encoded;
- 15 repeated header rows embedded in the source data;
- 80,000 cleaned production-test records;
- 866 production lots;
- date range: 4 October 2019 to 20 December 2019;
- 78,932 passing records and 1,068 defective records;
- overall observed defective rate: 1.335%;
- 19 continuous electrical measurements: `F1` through `F19`;
- among lots with at least 100 observations, the 95th-percentile defect rate is 5.36% and the maximum is 8.77%.

Those figures are used as data-integrity checks, not as substitutes for the raw records.

## Primary tasks

1. **Defect risk scoring:** predict whether a production test is defective from `F1..F19`.
2. **Lot-safe evaluation:** keep every production lot entirely inside one CV fold to prevent lot leakage.
3. **Imbalance-aware model selection:** optimize average precision/PR-AUC rather than raw accuracy because defects are only about 1.3% of records.
4. **Lot quality monitoring:** aggregate defect rate, Wilson confidence intervals, and model risk by lot.
5. **Adaptive inspection support:** flag stable lots whose observed defect rate crosses a configurable warning threshold.

## Why random row splitting is rejected

Records from the same manufacturing lot can share supplier, material, setup, process, fixture, and environmental effects. A random row split can therefore leak lot-specific signatures into both train and validation sets. The default pipeline uses `StratifiedGroupKFold` with `LOT` as the group.

For a deployment study, a second evaluation should also use a strict forward-in-time holdout to estimate performance under process drift. The lot-grouped CV in this repository is the baseline model-selection protocol; it is not a claim that the process is temporally stationary.

## Data handling

```bash
cd quality-monitoring/electrical-test-report
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/validate_data.py
```

The raw Kaggle CSV is intentionally not committed because the dataset redistribution license was not independently verified. The downloader retrieves the source at use time and creates a local SHA-256 manifest.

## Train

```bash
python -m src.train --config config.yaml
```

The training pipeline compares class-balanced logistic regression, random forest, and extra-trees models using lot-grouped out-of-fold predictions. The primary selection metric is average precision. It then chooses an operating threshold that meets the configured recall target while maximizing precision.

Generated outputs:

```text
reports/cv_metrics.json
reports/oof_predictions.csv
reports/lot_quality_summary.csv
reports/feature_importance.csv
artifacts/defect_model.joblib
```

## Lot monitoring

```bash
python -m src.monitoring data/raw/Electrical_Test_Report.csv \
  --min-lot-size 100 \
  --warning-rate 0.05
```

This produces a lot-level quality table with sample size, observed defects, defect rate, 95% Wilson interval, stability flag, and warning flag. The 5% threshold is an operational example derived from the visible upper tail in the published lot distribution; it should be calibrated against plant economics and quality requirements in a real deployment.

## Repository layout

```text
electrical-test-report/
├── README.md
├── SOURCES.md
├── config.yaml
├── requirements.txt
├── Makefile
├── data/
│   ├── README.md
│   └── raw/
├── notebooks/
│   └── 01_exploration.ipynb
├── reports/
│   └── README.md
├── scripts/
│   ├── download_data.py
│   └── validate_data.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   ├── monitoring.py
│   └── train.py
└── tests/
    ├── test_data.py
    ├── test_modeling.py
    └── test_monitoring.py
```

## Industrial interpretation

The useful Industry 4.0 question is not merely "can a classifier reproduce the `Result` column?" The higher-value question is whether the electrical signature can provide an early or independent risk signal, whether quality deteriorates by lot, and whether inspection effort should be intensified selectively for higher-risk lots rather than applied uniformly.

Because channel semantics are anonymized, feature importance must be interpreted as *diagnostic prioritization*, not physical causality. A plant deployment would require the test bench specification that maps `F1..F19` to actual electrical quantities and engineering limits.
