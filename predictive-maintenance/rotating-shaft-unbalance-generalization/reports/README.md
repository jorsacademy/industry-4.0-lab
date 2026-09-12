# Reports

Derived benchmark reports are written here by `python -m src.train --config config.yaml`.

The verified experiment selects the model family and representation using only the five development (`D`) recordings, then evaluates the frozen selection once on the five separately acquired evaluation (`E`) recordings.

The current `E` session has been consumed and is **closed for further model, feature, calibration, or sensor selection**. `rpm_band_performance.csv` and `sensor_ablation.csv` are post-selection diagnostics only. Any revised method requires a new independent session or a different pre-registered benchmark before an improvement claim is valid.

Primary files:

- `model_selection.csv` — blocked development-only comparison;
- `metrics.json` — frozen external-session result and evaluation contract;
- `evaluation_predictions.csv` — per-window `E` predictions and probabilities;
- `class_report.csv` and `confusion_matrix.csv` — severity-level external diagnostics;
- `rpm_band_performance.csv` — post-selection RPM-regime diagnostics;
- `sensor_ablation.csv` — post-selection sensor diagnostics;
- `window_summary.csv` — D/E recording coverage and RPM ranges;
- `feature_importance.csv` — selected-family feature importance fitted on development data.
