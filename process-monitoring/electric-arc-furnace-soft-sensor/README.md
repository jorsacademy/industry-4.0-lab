# Electric Arc Furnace — Leakage-Safe Terminal Temperature Soft Sensor

An Industry 4.0 process-monitoring project built on real electric-arc-furnace production data. The source links each steel heat to timestamped temperature/oxidation measurements, transformer operation, oxygen and gas usage, carbon injection, and material charging events.

The operational question is deliberately narrower than generic furnace forecasting:

> At the penultimate temperature measurement, can the final EAF temperature be forecast using only process information that was already available at that snapshot?

This framing creates a practical soft-sensor benchmark while making the prediction timestamp explicit.

## Why it adds something new

The repository already contains machine-condition, smart-quality and machining-response projects. This project adds a different cyber-physical process: batch metallurgy with heterogeneous event logs and heat-level traceability.

The source includes `HEATID` keys across process tables, timestamped transformer stages, gas/oxygen and carbon-injection streams, material additions, and repeated temperature/oxidation measurements. The current v1 benchmark intentionally uses a compact snapshot-safe core of temperature/oxidation, transformer and charging tables. Higher-rate gas/oxygen and carbon streams remain a documented extension rather than being silently mixed into the first benchmark.

## Leakage-safe snapshot design

For every heat with at least two valid temperature measurements:

1. the **target** is the final recorded EAF temperature;
2. the **snapshot time** is the preceding temperature measurement;
3. the latest measured temperature and oxidation available at the snapshot are retained;
4. transformer and material-charge events are aggregated only when their timestamps are at or before the snapshot;
5. events after the snapshot are never used as features.

This avoids a common industrial-data error: predicting an end-of-heat property with process events that happened after the nominal prediction point.

## Evaluation protocol

Heats are ordered by final-measurement time and divided chronologically into four blocks:

- fit — candidate-model fitting;
- selection — model-family selection;
- calibration — conformal absolute-residual calibration;
- final future test — used once for the reported generalization result.

A persistence baseline predicts that the final temperature will equal the penultimate measured temperature. Learned models must beat that operationally meaningful baseline, not merely a global mean.

The selected model is evaluated with MAE, RMSE, R², bias, and the fraction of heats within ±10 °C and ±20 °C. A split-conformal interval is calibrated on the later calibration block and frozen before the final future test.

## Feature families

The v1 benchmark uses only information observable by the snapshot:

- penultimate measured temperature and latest available positive oxidation reading;
- number of earlier temperature observations and forecast horizon;
- elapsed process time inferred from the earliest timestamped event for the heat;
- transformer segment count, stage, power-value summaries and duration summaries;
- basket-charge and additional-charge totals/counts.

The project intentionally avoids the final EAF chemical table as a predictor because its timing relative to the target can be ambiguous and would weaken the leakage guarantee. The high-frequency gas/oxygen and injected-carbon streams are also excluded from v1 until they are integrated through a separately tested streaming aggregation path with the same snapshot cutoff.

## Verified chronological benchmark

The benchmark workflow downloaded the public source and built **18,303** leakage-safe heat snapshots spanning January 2015 through July 2018. The median prediction horizon from the penultimate to final temperature measurement is **1 minute**. Seventeen numeric snapshot features are used.

Model selection on the chronological selection block chose **Extra Trees** over Random Forest and Ridge. The untouched final block contains **2,746 later heats** (November 2017 through July 2018).

| Metric | Extra Trees | Persistence baseline |
|---|---:|---:|
| MAE | 9.33 °C | 22.12 °C |
| RMSE | 12.28 °C | 28.62 °C |
| R² | 0.096 | -3.912 |
| Within ±10 °C | 61.8% | 33.1% |
| Within ±20 °C | 89.7% | 54.8% |

The learned soft sensor reduces final-test MAE by **57.8%** relative to simply carrying the penultimate temperature forward. The modest final-test R² is retained rather than hidden: the model is useful as a short-horizon correction to persistence, but it does not explain most of the cross-heat temperature variance.

The 90% split-conformal target produced a frozen interval radius of **17.79 °C** and achieved **85.7%** coverage on the future block. This under-coverage is reported explicitly; the interval should not be treated as a calibrated safety bound under the later regime.

The final chronological block has now been consumed and is closed for model/threshold tuning. A materially changed feature set—particularly one adding high-rate oxygen/gas or carbon streams—needs a new independent evaluation period or external EAF dataset before an improvement claim is made.

## Reproduce

```bash
cd process-monitoring/electric-arc-furnace-soft-sensor
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
reports/test_predictions.csv
reports/feature_importance.csv
reports/snapshot_summary.csv
artifacts/eaf_temperature_model.joblib
```

The raw Kaggle files are downloaded transiently and are not committed here. The source dataset is published under the MIT license.

## Interpretation boundary

This is observational industrial process data. Feature importance is not a causal statement about furnace physics, and the model should not be used to recommend transformer or charge interventions without process-engineering validation. The conformal interval is an empirical uncertainty estimate under the observed historical regime, not a safety guarantee under major regime change.
