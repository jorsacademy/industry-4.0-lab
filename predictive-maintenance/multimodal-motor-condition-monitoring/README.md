# Multimodal Motor Condition Monitoring — Acoustic + Vibration Diagnosis

An Industry 4.0 condition-monitoring project built on measured structure-borne vibration and airborne sound from an induction-motor / air-compressor demonstrator. Eight operating conditions are physically configured on the rig, including shaft unbalance, capacitor deactivation, outlet restriction, fan-housing clogging, and a defective fan.

The project adds a modality that is not covered by the repository's CNC controller telemetry: **3-axis vibration + microphone acoustics**. It also treats the source dataset's overlap structure as a first-class validation problem rather than reporting a random-window classifier score.

## Why this dataset is useful — and where it is limited

The source is measured, not synthetic. A 3-axis accelerometer is sampled on the motor structure and a microphone records airborne sound. The published feature table contains 2,000 short-time Fourier transform windows: 250 windows for each of eight operating conditions, with 69 structure-borne frequency amplitudes and 100 airborne-sound amplitudes.

However, the 250 windows for each condition come from **one 10-second recording**, and the published STFT uses 200 ms windows with **80% overlap**. Randomly shuffling those 2,000 rows into train and test sets would leak near-duplicate signal content across the split and create an unrealistically easy benchmark.

This repository therefore does not claim independent-machine or independent-session fault-diagnosis performance from this dataset.

## Evaluation protocol

### Purged blocked cross-validation

For every condition, each fold holds out one contiguous time block. Training windows within four STFT hops of the test block are removed. With a 200 ms window and 80% overlap, this purge prevents the raw signal support of adjacent training windows from directly overlapping the held-out window support.

The protocol is intentionally described as **same-session blocked CV**. It measures temporal robustness inside the published recordings, not repeatability across new recording sessions, sensor mountings, ambient environments, machines, or maintenance events.

### Modality ablation

The benchmark compares three sensor configurations:

1. vibration only — 3-axis structure-borne frequency features;
2. audio only — microphone frequency features;
3. early fusion — both modalities.

Each modality is evaluated with logistic regression, Random Forest and Extra Trees. Model selection uses macro-F1 first and balanced accuracy second so that all eight operating conditions matter.

### Diagnostic outputs

The project commits the full model/modality comparison, blocked out-of-fold predictions, confusion matrix and per-class report. The fitted full-data model is uploaded as a workflow artifact rather than committed as a binary.

## Reproduce

```bash
cd predictive-maintenance/multimodal-motor-condition-monitoring
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
reports/model_modality_comparison.csv
reports/blocked_oof_predictions.csv
reports/confusion_matrix.csv
reports/class_report.csv
reports/metrics.json
artifacts/condition_model.joblib
```

## Interpretation boundary

A high same-session score can still coexist with poor real deployment performance. The next scientifically valid step is to collect repeated sessions for every condition across different days, sensor remounts, loads and preferably multiple machines. Those sessions should become the grouping unit for a true external generalization benchmark.

Until that data exists, this project is best interpreted as a **leakage-aware multimodal signal-diagnostics benchmark**, not a production predictive-maintenance claim.
