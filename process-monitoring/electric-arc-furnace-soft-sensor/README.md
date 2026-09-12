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
