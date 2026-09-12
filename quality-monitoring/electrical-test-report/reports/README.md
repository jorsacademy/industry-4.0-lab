# Generated reports

Training and monitoring commands write reproducible outputs here. Expected files include:

- `cv_metrics.json`: grouped out-of-fold model metrics and the selected operating threshold;
- `oof_predictions.csv`: record-level out-of-fold defect probabilities;
- `lot_quality_summary.csv`: lot-level observed defect rates, Wilson intervals, and model risk;
- `feature_importance.csv`: model-specific ranking of anonymized electrical measurements.

Generated report CSV/JSON files are intentionally not committed by default.
