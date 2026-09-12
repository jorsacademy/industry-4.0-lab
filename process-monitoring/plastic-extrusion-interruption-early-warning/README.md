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

This rule was fixed from source semantics before model benchmarking. In the verified source inspection it yields **193,941 active rows and 640 interruption-proxy onsets** across the published sequence, with events present in every chronological evaluation block.

The proxy can include planned/operator stops as well as unplanned disruptions. It is therefore an early-warning benchmark for production interruption, not a causal equipment-failure detector.

## Leakage-safe 20-minute warning contract

1. The warning horizon is fixed at **20 minutes** before model selection.
2. Only currently active-production rows are scored.
3. A row is positive when the next proxy onset occurs within 20 minutes in the same continuous timestamp segment.
4. Segments are broken at source gaps greater than five minutes, so labels never bridge long data outages.
5. Current layer-thickness values, current actual extrusion outputs and the controls used to define the active-state filter are excluded from model predictors.
6. Train-only filtering removes high-missingness and constant variables.
7. Feature ranking is fit only on the historical fit block.
8. Fit, model/panel selection, threshold calibration and final future test are chronological and separated by purge windows longer than the warning horizon.
9. The final future block is opened once. It is not used to redesign the model or threshold.

## Candidate models

The development search compares:

- class-weighted logistic regression;
- class-weighted Extra Trees;
- compact train-ranked panels of 40, 80 and 120 variables plus the full eligible panel.

Average precision (PR-AUC) is the primary selection metric because the upcoming-interruption target is rare. Among candidates within 0.01 absolute PR-AUC of the best selection result, the smallest panel is preferred.

The selected model/panel is then refit on the combined fit + selection history. A frozen operating threshold is chosen on the later calibration block by maximizing precision subject to recall of at least 0.70 when feasible.

## Diagnostics

The final benchmark reports:

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

After the verified benchmark is run, its final future block is treated as consumed and closed. Any method redesigned in response to that final result requires a new independent period or a separately pre-registered benchmark for a valid improvement claim.
