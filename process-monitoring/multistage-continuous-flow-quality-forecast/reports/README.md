# Reports

Derived aggregate benchmark reports are written here by:

```bash
python -m src.train --config config.yaml
```

Per-row raw targets or reconstructed source tables are not committed because the source data are not distributed under a permissive license.

The verified experiment uses historical fit/model-selection blocks, a later conformal-calibration block, and one final future block. The final future block has been consumed and is **closed for further horizon, feature, model-family, calibration, or sensor selection**.

Primary files:

- `model_selection.csv` — historical-only horizon/model comparison and pre-test baselines;
- `metrics.json` — frozen final benchmark result, split ranges, baselines, bootstrap comparison, and uncertainty summary;
- `per_target_metrics.csv` — aggregate error and coverage diagnostics for each of the 15 Stage-2 outputs;
- `interval_coverage.csv` — marginal split-conformal coverage by target;
- `drift_diagnostics.csv` — fit-to-future feature-distribution shift diagnostics;
- `feature_importance.csv` — observational importance from the selected learned model.

The verified result is negative in operational terms: the process-only learned model is substantially worse than current-output persistence on the future block, and the nominal 90% conformal intervals under-cover. Those results are retained rather than tuned away. Any improvement claim requires a new independent run or a different pre-registered benchmark.
