from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import read_feature_table
from .modeling import (
    PlattCalibrator,
    classification_metrics,
    feature_columns,
    fit_and_select_model,
    select_operating_threshold,
    temporal_four_way_split,
)


def run_training(config_path: Path, feature_path: Path | None = None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    data_cfg = cfg["data"]
    model_cfg = cfg["modeling"]
    split_cfg = cfg["split"]
    report_cfg = cfg["reports"]

    if feature_path is None:
        prefix = float(data_cfg.get("prefix_fraction", 1.0))
        tag = f"p{int(round(prefix * 100)):03d}"
        feature_path = Path(data_cfg["processed_dir"]) / f"bosch_features_{tag}.parquet"

    frame = read_feature_table(feature_path)
    if "Response" not in frame:
        raise ValueError("Training feature table has no Response column")
    split = temporal_four_way_split(
        frame,
        train_fraction=float(split_cfg["train_fraction"]),
        selection_fraction=float(split_cfg["selection_fraction"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )

    candidates = list(model_cfg.get("candidates", ["logistic", "hist_gradient"]))
    random_state = int(model_cfg.get("random_state", 42))
    selected_name, model, comparison = fit_and_select_model(frame, split, candidates, random_state)

    columns = feature_columns(frame)
    x = frame[columns]
    y = frame["Response"].astype(int).to_numpy()

    raw_calibration = model.predict_proba(x.iloc[split.calibration])[:, 1]
    calibrator = PlattCalibrator().fit(raw_calibration, y[split.calibration])
    calibrated_calibration = calibrator.predict(raw_calibration)

    threshold_cfg = model_cfg.get("operating_threshold", {})
    threshold = select_operating_threshold(
        y[split.calibration],
        calibrated_calibration,
        strategy=str(threshold_cfg.get("strategy", "min_precision")),
        min_precision=float(threshold_cfg.get("min_precision", 0.10)),
    )

    raw_test = model.predict_proba(x.iloc[split.test])[:, 1]
    calibrated_test = calibrator.predict(raw_test)
    metrics = classification_metrics(y[split.test], calibrated_test, threshold)

    reports_dir = Path(report_cfg.get("directory", "reports"))
    artifacts_dir = Path(report_cfg.get("artifacts_directory", "artifacts"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    prediction_frame = pd.DataFrame(
        {
            "Id": frame.iloc[split.test]["Id"].to_numpy(),
            "start_time": frame.iloc[split.test]["start_time"].to_numpy(),
            "Response": y[split.test],
            "raw_probability": raw_test,
            "calibrated_probability": calibrated_test,
            "prediction": (calibrated_test >= threshold).astype(int),
        }
    )
    prediction_frame.to_csv(reports_dir / "test_predictions.csv", index=False)

    result = {
        "feature_table": str(feature_path),
        "rows": int(len(frame)),
        "prevalence": float(y.mean()),
        "selected_model": selected_name,
        "partitions": {
            "train": int(len(split.train)),
            "selection": int(len(split.selection)),
            "calibration": int(len(split.calibration)),
            "test": int(len(split.test)),
        },
        "test_metrics": metrics,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    if selected_name == "logistic":
        fitted_imputer = model.named_steps["imputer"]
        fitted_model = model.named_steps["model"]
        transformed_names = list(fitted_imputer.get_feature_names_out(columns))
        coefficients = np.abs(fitted_model.coef_[0])
        pd.DataFrame({"feature": transformed_names, "abs_coefficient": coefficients}).sort_values(
            "abs_coefficient", ascending=False
        ).to_csv(reports_dir / "feature_coefficients.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "threshold": threshold,
            "feature_columns": columns,
            "selected_model": selected_name,
        },
        artifacts_dir / "bosch_quality_model.joblib",
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--features", type=Path, default=None)
    args = parser.parse_args()
    run_training(args.config, args.features)


if __name__ == "__main__":
    main()
