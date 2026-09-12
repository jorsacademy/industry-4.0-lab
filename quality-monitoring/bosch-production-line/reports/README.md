# Reports

No benchmark metric is committed until the authorized Bosch competition data has been obtained and the pipeline has actually run.

A full local training run writes:

```text
reports/model_comparison.csv
reports/metrics.json
reports/test_predictions.csv
reports/feature_coefficients.csv   # when the selected model exposes linear coefficients
```

An early-warning study writes:

```text
reports/early_warning_tradeoff.csv
```

The raw and processed competition data remain untracked.
