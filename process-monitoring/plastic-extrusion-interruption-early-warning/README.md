# Plastic Extrusion — Active-Line Interruption Early Warning

A leakage-aware Industry 4.0 process-monitoring benchmark built on a year of real plastic-extrusion telemetry. The source contains **226,536 timestamped rows and 470 columns** spanning extruder, haul-off, winding, thickness, output, temperature, pressure and other line variables.

The project asks a deliberately narrow operational question:

> While the line is observably producing, can process telemetry available now rank the risk of an active-production interruption within the next 20 minutes?

## Target definition: an interruption proxy, not a failure label

The source does **not** contain an explicit fault-cause or film-break label. Its legend identifies actual layer-thickness channels, actual output channels, haul-off state/speed and winder-running signals, but no authoritative stop-cause field.

For that reason this project does not call every zero-thickness episode a film break. It defines an operational **active-line interruption proxy** as a transition from an active-production row to a row where all four measured layer-thickness channels are zero, provided the timestamp gap is at most three minutes.

A row is considered actively producing only when all of the following are already true:

- all four actual layer-thickness values are positive;
- all four actual extrusion-output values are positive;
- haul-off 1 is enabled and closed;
- haul-off set speed is at least 10;
- at least one of the two winding stations is running.

This rule was fixed from source semantics before model benchmarking. It yields **193,941 active rows and 640 interruption-proxy onsets** across the published sequence. The proxy can include planned/operator stops as well as unplanned disruptions, so this is an early-warning benchmark for production interruption rather than a causal equipment-failure detector.

## Leakage-safe 20-minute warning contract

1. The warning horizon is fixed at **20 minutes** before model selection.
2. Only currently active-production rows are scored.
3. A row is positive when the next proxy onset occurs within 20 minutes in the same continuous timestamp segment.
4. Segments are broken at source gaps greater than five minutes, so labels never bridge long data outages.
5. Current layer-thickness values, current actual extrusion outputs and the controls used to define the active-state filter are excluded from model predictors.
6. Train-only filtering removes high-missingness and constant variables.
7. Feature ranking is fit only on the historical fit block.
8. Fit, model/panel selection, threshold calibration and final future test are chronological and separated by 30-minute purge windows.
9. The final future block is opened once. It is not used to redesign the model or threshold.

## Candidate models

The development search compares class-weighted logistic regression and Extra Trees on train-ranked panels of 40, 80, 120 and all eligible variables. Average precision (PR-AUC) is the primary selection metric. Among candidates within 0.01 absolute PR-AUC of the best selection result, the smallest panel is preferred.

The selected model/panel is refit on the combined fit + selection history. A frozen operating threshold is chosen on the later calibration block by maximizing precision subject to recall of at least 0.70 when feasible.

## Verified benchmark

The validated source spans **2018-06-25 to 2019-06-25** and contains 23 continuous timestamp segments after splitting at gaps over five minutes. After direct-state exclusion there are 456 numeric predictor candidates; train-only missingness/constant filtering leaves 317 eligible variables.

Historical selection chose **Extra Trees with an 80-variable panel**:

| Metric | Selection block |
|---|---:|
| PR-AUC | **0.1684** |
| ROC-AUC | 0.7133 |
| Brier score | 0.0405 |

The untouched future block contains **29,583 active rows**, of which 874 are within 20 minutes of an interruption-proxy onset, for a base prevalence of **2.95%**. The frozen model produces:

| Metric | Future block |
|---|---:|
| PR-AUC | **0.1607** |
| ROC-AUC | 0.7085 |
| Brier score | 0.0414 |
| Precision | 0.0406 |
| Recall | 0.7872 |
| F1 | 0.0772 |
| MCC | 0.0755 |
| Balanced accuracy | 0.6102 |

PR-AUC is about **5.44× the future base prevalence**, so the telemetry contains useful ranking signal. The frozen operating threshold, however, is not deployment-ready: it raises alarms on **57.3% of active rows** and produces 16,269 false positives for 688 true positives. High recall therefore comes at an operationally unacceptable alarm burden.

At the event level, 96 of 104 eligible future interruption-proxy onsets receive at least one warning in the full 20-minute window (**92.3% event recall**), with a median first-warning lead of **19 minutes**. This should be interpreted together with the very high row-level false-alarm rate, not as a standalone success metric.

The selected panel also exhibits substantial temporal drift. Two haul-off hour-counter variables have KS statistic `1.0`, and several controller/process variables show large distribution changes. These diagnostics establish regime shift; they do not identify causal reasons for interruption.

The final future block is now **consumed and closed**. The model, variable panel, warning horizon and threshold will not be revised against these final metrics and then reported on the same block as untouched. A valid improvement requires a new independent production period or another pre-registered benchmark.

## Diagnostics

The committed aggregate reports contain:

- PR-AUC, ROC-AUC and Brier score;
- precision, recall, F1, MCC and balanced accuracy;
- confusion counts and alarm rate per 1,000 active rows;
- event-level warning recall and first-warning lead time;
- selected-panel drift from fit to future test using KS statistics and robust median shifts;
- aggregate feature importance.

No feature-importance result is interpreted as a causal process intervention.

## Source and redistribution boundary

Source: `podsyp/find-a-defect-in-the-production-extrusion-line` on Kaggle.

License: **CC BY-NC-ND 4.0**. Raw data, transformed row-level data and fitted model binaries are not redistributed by this repository. The benchmark workflow downloads the public source transiently for reproducible analysis. Only code and aggregate reports are committed.

## Reproduce

```bash
cd process-monitoring/plastic-extrusion-interruption-early-warning
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
```

## Interpretation boundary

This is observational process telemetry and the target is an operational interruption proxy. It does not identify why production stopped, and it cannot distinguish a planned stop from an unplanned break without an authoritative stop-cause label. A useful ranking signal can support operator attention or deeper process investigation; it does not establish that changing any selected variable would prevent an interruption.
