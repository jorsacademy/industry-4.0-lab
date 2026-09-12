# Generated reports

After `python -m src.train --config config.yaml`:

- `panel_model_comparison.csv` — validation metrics across model and process-variable panel sizes;
- `metrics.json` — selected configuration, chronological block sizes/class counts and final future-test metrics;
- `selected_process_variables.csv` — selected anonymous variables, rank, fit-history missingness and screening score;
- `drift_report.csv` — selected-variable missingness and robust median shifts between history and future test;
- `test_predictions.csv` — final-test probabilities and fixed operating decision.

Generated benchmark reports may be committed by the dedicated GitHub Actions benchmark workflow. Model binaries remain workflow artifacts.
