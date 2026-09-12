from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .config import load_config, project_path
from .data import load_all_experiments
from .features import build_feature_table
from .modeling import (
    candidate_models,
    choose_best,
    encode_binary_target,
    evaluate_pipeline,
    extract_feature_importance,
    feature_columns,
)


def train(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    data_cfg = config["data"]
    feature_cfg = config["features"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    raw_dir = project_path(config, data_cfg["raw_dir"])
    _, experiments = load_all_experiments(
        raw_dir,
        metadata_file=data_cfg["metadata_file"],
        pattern=data_cfg["experiment_glob"],
        mask_known_artifacts=bool(data_cfg["mask_known_artifact_values"]),
    )

    target = str(model_cfg["target"])
    feature_table = build_feature_table(
        experiments,
        target=target,
        window_size=int(feature_cfg["window_size"]),
        step_size=int(feature_cfg["step_size"]),
        minimum_rows_per_window=int(feature_cfg["minimum_rows_per_window"]),
        statistics=list(feature_cfg["statistics"]),
        sample_period_seconds=float(data_cfg["sample_period_seconds"]),
    )

    y = encode_binary_target(feature_table["target"], str(model_cfg["positive_label"]))
    groups = feature_table["experiment_id"].astype(int)
    numeric, categorical = feature_columns(feature_table)
    candidates = candidate_models(numeric, categorical, int(model_cfg["random_state"]))

    results = []
    for name, pipeline in candidates.items():
        results.append(
            evaluate_pipeline(
                name,
                pipeline,
                feature_table,
                y,
                groups,
                maximum_splits=int(model_cfg["maximum_cv_splits"]),
                random_state=int(model_cfg["random_state"]),
                threshold=float(model_cfg["probability_threshold"]),
            )
        )

    best = choose_best(results)
    X = feature_table[numeric + categorical]
    final_pipeline = candidates[best.model_name]
    final_pipeline.fit(X, y)

    artifact_dir = project_path(config, output_cfg["artifact_dir"])
    report_dir = project_path(config, output_cfg["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / output_cfg["model_file"]
    joblib.dump(final_pipeline, model_path)

    all_metrics = {result.model_name: result.metrics for result in results}
    metrics_payload: dict[str, object] = {
        "selected_model": best.model_name,
        "target": target,
        "positive_label": model_cfg["positive_label"],
        "probability_threshold": float(model_cfg["probability_threshold"]),
        "window_size": int(feature_cfg["window_size"]),
        "step_size": int(feature_cfg["step_size"]),
        "models": all_metrics,
    }
    (report_dir / "cv_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    best.window_predictions.to_csv(report_dir / "oof_window_predictions.csv", index=False)
    best.run_predictions.to_csv(report_dir / "oof_run_predictions.csv", index=False)
    extract_feature_importance(final_pipeline).head(100).to_csv(
        report_dir / "feature_importance.csv", index=False
    )

    metadata_payload = {
        "selected_model": best.model_name,
        "target": target,
        "positive_label": model_cfg["positive_label"],
        "probability_threshold": float(model_cfg["probability_threshold"]),
        "feature_columns_numeric": numeric,
        "feature_columns_categorical": categorical,
        "window_size": int(feature_cfg["window_size"]),
        "step_size": int(feature_cfg["step_size"]),
        "minimum_rows_per_window": int(feature_cfg["minimum_rows_per_window"]),
        "statistics": list(feature_cfg["statistics"]),
        "sample_period_seconds": float(data_cfg["sample_period_seconds"]),
        "mask_known_artifact_values": bool(data_cfg["mask_known_artifact_values"]),
    }
    (artifact_dir / output_cfg["metadata_file"]).write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe CNC tool-wear models.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    payload = train(args.config)
    selected = payload["selected_model"]
    run_metrics = payload["models"][selected]["run_level"]
    print(json.dumps({"selected_model": selected, "run_level": run_metrics}, indent=2))


if __name__ == "__main__":
    main()
