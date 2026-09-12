# Rotating-Shaft Unbalance — Independent-Session Generalization

A leakage-aware rotating-machinery condition-monitoring benchmark built on measured vibration from a variable-speed drive train. The source provides five unbalance severities and, critically, a separate **development (`D`) recording and evaluation (`E`) recording for every severity**.

The project is designed around that separation. Model family and representation are selected only on the development recordings. The evaluation recordings were opened once after selection for the verified benchmark below and are now closed to further model or sensor selection.

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
- development RPM range observed by this benchmark: approximately 629–2355 RPM
- evaluation RPM range observed by this benchmark: approximately 1048–1960 RPM

The ten source recordings are `0D..4D` and `0E..4E`. Severity 0 is the unbalance-free holder. Severities 1–4 use progressively stronger physical unbalances defined by attached mass/radius combinations.

Raw source files are not committed to this repository. The benchmark workflow downloads the canonical archive transiently and stores a SHA-256 manifest.

## Evaluation contract

### 1. Independent-session boundary

- `D` recordings: feature engineering plus model-family and representation selection.
- `E` recordings: one final external-session evaluation after selection is frozen.
- RPM-band and sensor-ablation analyses on `E` are post-selection diagnostics only; they cannot be used to replace the selected model or claim an improved final result.

The `E` session has now been consumed. Any revised feature set, model family, calibration rule, or sensor-selection policy requires a new independent session or a different pre-registered benchmark for a valid improvement claim.

### 2. Non-overlapping windows

The raw signal is converted into 4-second windows with a 20-second hop. This deliberately avoids overlapping-window leakage and reduces serial dependence relative to dense sliding-window classification.

The verified run contains **1,608 development windows** and **418 evaluation windows**.

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

## Verified benchmark

Development-only blocked validation selected **Extra Trees on the 44 compact physics-informed features**. Its development result was:

| Metric | Development blocked CV |
|---|---:|
| Macro-F1 | **0.9845** |
| Macro-F1 standard deviation | 0.0184 |
| Balanced accuracy | 0.9845 |

That result did **not** transfer to the independently acquired `E` recordings:

| Metric | External `E` session |
|---|---:|
| Macro-F1 | **0.4263** |
| Balanced accuracy | 0.4621 |
| Accuracy | 0.4617 |
| Ordinal severity MAE | 0.7775 |
| Multiclass Brier score | 0.8603 |
| Log loss | 2.0037 |
| Expected calibration error | 0.3977 |

The large `D`→`E` gap is the central result of the project. A model that appears almost solved under blocked validation inside the development recordings can fail badly when the acquisition session changes, even on the same laboratory drive train.

The class-level failure is not uniform. Severity 4 transfers strongly (`F1 = 0.9880`), while severity 1 has `F1 = 0.0000`; severities 2 and 3 are also substantially weaker. This indicates that the external-session problem is concentrated in light and intermediate unbalance states rather than being a simple across-the-board loss of signal.

RPM diagnostics also show weaker transfer in the upper evaluation-speed band: macro-F1 falls to **0.3340** for 1700–2000 RPM, versus 0.4996 below 1200 RPM. These are post-selection diagnostics, not alternative model-selection criteria.

A single-sensor post-selection diagnostic found sensor 1 alone at macro-F1 `0.4749`, above the all-sensor result of `0.4263`. Because this comparison was made after opening `E`, the project deliberately does **not** switch the final model to sensor 1 or report that value as a new selected benchmark.

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
reports/feature_importance.csv
artifacts/unbalance_model.joblib
```

## Interpretation boundary

The separate `D` and `E` recordings make this materially stronger than a random-window benchmark, but both sessions still come from the same laboratory drive train and sensor installation. External-session performance measures transfer across the two published acquisition sessions; it is not evidence of transfer to a different machine, bearing, mounting, sensor model or industrial environment.

The poor external result is retained rather than tuned away. The scientifically valid next step is a newly acquired independent session or another pre-registered rotating-machinery dataset. Re-optimizing against the consumed `E` recordings would turn the external benchmark into another development set.

Unbalance severity is an experimental condition, not a universal damage scale. The model is a diagnostic benchmark and should not be converted into maintenance thresholds without machine-specific validation.
