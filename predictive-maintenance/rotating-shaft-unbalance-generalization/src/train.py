from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config
from .data import build_feature_table
from .modeling import (
    evaluate_classifier,
    feature_importance_table,
    fit_calibrated,
    fit_uncalibrated,
    rpm_band_table,
    select_candidate,
    sensor_ablation_external,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root = config_path.parent
    cfg = load_config(config_path)
    archive = root / "data" / "raw" / cfg["source"]["archive_name"]
    if not archive.exists():
        raise FileNotFoundError(f"Missing source archive: {archive}. Run scripts/download_data.py first.")

    reports = root / "reports"
    artifacts = root / "artifacts"
    processed = root / "data" / "processed"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    n_folds = int(cfg["validation"]["development_folds"])
    feature_table = build_feature_table(archive, cfg["signal"], n_folds=n_folds)
    feature_table.to_csv(processed / "window_features.csv", index=False)

    dev = feature_table[feature_table["session"] == "D"].reset_index(drop=True)
    evaluation = feature_table[feature_table["session"] == "E"].reset_index(drop=True)
    if set(dev["severity"].unique()) != set(range(5)) or set(evaluation["severity"].unique()) != set(range(5)):
        raise ValueError("Both D and E must contain all five severities")

    random_state = int(cfg["modeling"]["random_state"])
    n_jobs = int(cfg["modeling"]["n_jobs"])
    bins = int(cfg["modeling"]["calibration_bins"])
    selected, selection = select_candidate(dev, n_folds=n_folds, random_state=random_state, n_jobs=n_jobs)
    selection.to_csv(reports / "model_selection.csv", index=False)

    calibrated, cols = fit_calibrated(dev, selected, n_folds=n_folds)
    metrics, predictions, matrix, class_report = evaluate_classifier(
        calibrated, evaluation, cols, calibration_bins=bins
    )
    predictions.to_csv(reports / "evaluation_predictions.csv", index=False)
    matrix.to_csv(reports / "confusion_matrix.csv")
    class_report.to_csv(reports / "class_report.csv", index=False)
    rpm = rpm_band_table(predictions, cfg["rpm_bands"])
    rpm.to_csv(reports / "rpm_band_performance.csv", index=False)

    ablation = sensor_ablation_external(
        dev, evaluation, selected, n_folds=n_folds, calibration_bins=bins
    )
    ablation.to_csv(reports / "sensor_ablation.csv", index=False)

    uncalibrated = fit_uncalibrated(dev, selected, cols)
    importance = feature_importance_table(uncalibrated, cols)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    summary = (
        feature_table.groupby(["session", "recording"], as_index=False)
        .agg(
            windows=("severity", "size"),
            severity=("severity", "first"),
            rpm_min=("compact__rpm_median", "min"),
            rpm_median=("compact__rpm_median", "median"),
            rpm_max=("compact__rpm_median", "max"),
        )
    )
    summary.to_csv(reports / "window_summary.csv", index=False)

    selected_row = selection.iloc[0].to_dict()
    result = {
        "source": {
            "development_windows": int(len(dev)),
            "evaluation_windows": int(len(evaluation)),
            "sample_rate_hz": int(cfg["signal"]["sample_rate_hz"]),
            "window_seconds": float(cfg["signal"]["window_seconds"]),
            "hop_seconds": float(cfg["signal"]["hop_seconds"]),
            "development_recordings": 5,
            "evaluation_recordings": 5,
        },
        "selection_contract": "model family and representation selected only from D using five blocked time folds; E opened once after selection",
        "selected": {
            "model": selected.name,
            "representation": selected.representation,
            "n_features": int(len(cols)),
            "dev_cv_macro_f1": float(selected_row["macro_f1"]),
            "dev_cv_macro_f1_std": float(selected_row["macro_f1_std"]),
            "dev_cv_balanced_accuracy": float(selected_row["balanced_accuracy"]),
        },
        "external_evaluation": {key: float(value) for key, value in metrics.items()},
        "rpm_range": {
            "development_min": float(dev["compact__rpm_median"].min()),
            "development_max": float(dev["compact__rpm_median"].max()),
            "evaluation_min": float(evaluation["compact__rpm_median"].min()),
            "evaluation_max": float(evaluation["compact__rpm_median"].max()),
        },
        "interpretation_note": (
            "E is a separately acquired session from the same laboratory drive train. "
            "External-session performance is not cross-machine generalization."
        ),
    }
    (reports / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"model": calibrated, "features": cols, "selected": selected.name}, artifacts / "unbalance_model.joblib")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
