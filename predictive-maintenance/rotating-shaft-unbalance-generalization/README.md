# Rotating-Shaft Unbalance — Independent-Session Generalization

A leakage-aware rotating-machinery condition-monitoring benchmark built on measured vibration from a variable-speed drive train. The source provides five unbalance severities and, critically, a separate **development (`D`) recording and evaluation (`E`) recording for every severity**.

The project is designed around that separation. Model selection is performed only on the development recordings. The evaluation recordings remain closed until the model family and sensor configuration are frozen.

## Why this project is different

The repository already contains a multimodal motor-diagnostics benchmark using vibration and microphone features. That dataset has one short recording per condition, so even purged blocked cross-validation cannot establish independent-session generalization.

This project addresses that exact limitation. It tests whether a model trained on one long set of variable-speed recordings transfers to separately acquired evaluation recordings under the same physical fault definitions but a different acquisition session and a narrower RPM regime.

## Source

Canonical raw CSV release:

- Fordatis / Fraunhofer: `Vibration Measurements on a Rotating Shaft at Different Unbalance Strengths`
- pinned raw release: `fordatis/151.2`, DOI `10.24406/fordatis/65.2`
- license: CC BY 4.0
- sampling rate: 4096 Hz
- channels: motor-controller input voltage, measured RPM, and three vibration sensors
- development RPM range: approximately 630–2330 RPM
- evaluation RPM range: approximately 1060–1900 RPM

The ten source recordings are `0D..4D` and `0E..4E`. Severity 0 is the unbalance-free holder. Severities 1–4 use progressively stronger physical unbalances defined by attached mass/radius combinations.

Raw source files are not committed to this repository. The benchmark workflow downloads the canonical archive transiently and stores a SHA-256 manifest.

## Evaluation contract

### 1. Independent-session boundary

- `D` recordings: feature engineering, model selection, hyperparameter/model-family comparison, sensor-ablation selection.
- `E` recordings: final external evaluation only.

No metric from `E` is used to choose the final model.

### 2. Non-overlapping windows

The raw signal is converted into 4-second windows with a 20-second hop. This deliberately avoids overlapping-window leakage and reduces serial dependence relative to dense sliding-window classification.

### 3. Physics-informed compact features

For each of the three vibration sensors the pipeline extracts:

- RMS, standard deviation, peak-to-peak range and crest factor;
- skewness and kurtosis;
- spectral entropy and spectral centroid;
- amplitudes around the 1×, 2× and 3× shaft orders;
- 1×-order energy relative to broadband spectral energy.

The shaft-order frequencies are computed from the measured RPM inside each window rather than using fixed-frequency bands.

### 4. Order-spectrum representation

A second representation samples each vibration spectrum at fractional shaft orders from 0.5× through 10×. This normalizes the spectral representation to rotational speed and allows a frequency-domain MLP to compete against classical compact-feature models.

### 5. Development-only model selection

Five blocked folds are constructed inside every `D` recording. Each fold holds out the same relative time segment from all five severity recordings. Candidate models include:

- multinomial logistic regression on compact features;
- Random Forest on compact features;
- Extra Trees on compact features;
- logistic regression on order-spectrum features;
- MLP on order-spectrum features.

Selection uses macro-F1 first and balanced accuracy second.

### 6. Final diagnostics

After freezing the selected model, the `E` recordings are evaluated with:

- macro-F1, balanced accuracy and overall accuracy;
- per-class precision/recall/F1;
- severity mean absolute error on the ordinal 0–4 labels;
- multiclass Brier score, log loss and expected calibration error;
- confusion matrix;
- performance by RPM band;
- post-selection sensor-ablation diagnostics using the same frozen model family.

The RPM-band and sensor-ablation tables are diagnostics. They do not retroactively change the selected model.

## Reproduce

```bash
cd predictive-maintenance/rotating-shaft-unbalance-generalization
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

The raw archive is large (about 2.6 GB compressed). Feature extraction streams CSV members directly from the ZIP; the ten raw CSV files are not extracted to disk.

## Outputs

```text
reports/model_selection.csv
reports/metrics.json
reports/evaluation_predictions.csv
reports/confusion_matrix.csv
reports/class_report.csv
reports/rpm_band_performance.csv
reports/sensor_ablation.csv
reports/window_summary.csv
artifacts/unbalance_model.joblib
```

## Interpretation boundary

The separate `D` and `E` recordings make this materially stronger than a random-window benchmark, but both sessions still come from the same laboratory drive train and sensor installation. External-session performance therefore supports repeatability across the published recording sessions; it is not evidence of transfer to a different machine, bearing, mounting, sensor model or industrial environment.

Unbalance severity is an experimental condition, not a universal damage scale. The model is a diagnostic benchmark and should not be converted into maintenance thresholds without machine-specific validation.
