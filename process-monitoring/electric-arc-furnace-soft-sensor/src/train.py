from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import build_snapshot_dataset, feature_columns, load_tables
from .modeling import (
    candidate_models,
    chronological_blocks,
    conformal_radius,
    interval_coverage,
    model_importance,
    regression_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    reports = ROOT / "reports"
    artifacts = ROOT / "artifacts"
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)

    tables = load_tables(raw_dir)
    dataset = build_snapshot_dataset(tables)
    features = feature_columns(dataset)
    if len(features) < 5:
        raise ValueError(f"Too few numeric snapshot features: {features}")

    ev = cfg["evaluation"]
    blocks = chronological_blocks(
        dataset,
        (ev["fit_fraction"], ev["selection_fraction"], ev["calibration_fraction"], ev["test_fraction"]),
    )

    model_cfg = cfg["models"]
    models = candidate_models(
        seed=int(cfg["seed"]),
        n_estimators=int(model_cfg["n_estimators"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        max_features=float(model_cfg["max_features"]),
        ridge_alpha=float(model_cfg["ridge_alpha"]),
    )

    X_fit = blocks["fit"][features]
    y_fit = blocks["fit"]["target_temp"]
    X_sel = blocks["selection"][features]
    y_sel = blocks["selection"]["target_temp"]

    comparisons: list[dict[str, object]] = []
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        pred = model.predict(X_sel)
        row = {"model": name, **regression_metrics(y_sel, pred)}
        comparisons.append(row)

    selection_table = pd.DataFrame(comparisons).sort_values(["mae", "rmse", "model"]).reset_index(drop=True)
    selected_name = str(selection_table.iloc[0]["model"])
    selected_model = models[selected_name]

    history = pd.concat([blocks["fit"], blocks["selection"]], ignore_index=True)
    selected_model.fit(history[features], history["target_temp"])

    cal = blocks["calibration"]
    cal_pred = selected_model.predict(cal[features])
    coverage_target = float(ev["interval_coverage"])
    radius = conformal_radius(cal["target_temp"], cal_pred, coverage_target)

    test = blocks["test"].copy()
    test_pred = selected_model.predict(test[features])
    test["prediction"] = test_pred
    test["lower"] = test_pred - radius
    test["upper"] = test_pred + radius
    test["persistence_prediction"] = test["snapshot_temp"].astype(float)
    test["error"] = test["prediction"] - test["target_temp"]

    learned_metrics = regression_metrics(test["target_temp"], test["prediction"])
    baseline_metrics = regression_metrics(test["target_temp"], test["persistence_prediction"])
    learned_metrics["interval_coverage"] = interval_coverage(test["target_temp"], test["lower"], test["upper"])
    learned_metrics["interval_radius_c"] = radius

    split_summary = {}
    for name, block in blocks.items():
        split_summary[name] = {
            "rows": int(len(block)),
            "start": block["target_time"].min().isoformat(),
            "end": block["target_time"].max().isoformat(),
        }

    summary = {
        "source_snapshot_rows": int(len(dataset)),
        "numeric_features": int(len(features)),
        "selected_model": selected_name,
        "interval_target_coverage": coverage_target,
        "splits": split_summary,
        "selection_metrics": selection_table.iloc[0].to_dict(),
        "future_test_metrics": learned_metrics,
        "persistence_baseline_metrics": baseline_metrics,
        "mae_improvement_vs_persistence": float((baseline_metrics["mae"] - learned_metrics["mae"]) / baseline_metrics["mae"])
        if baseline_metrics["mae"] > 0 else np.nan,
        "interpretation_note": "Snapshot features use only EAF events timestamped at or before the penultimate temperature measurement; observational feature importance is not causal process guidance.",
    }

    selection_table.to_csv(reports / "model_selection.csv", index=False)
    test[[
        "HEATID", "snapshot_time", "target_time", "snapshot_temp", "target_temp",
        "forecast_horizon_min", "prediction", "lower", "upper", "persistence_prediction", "error",
    ]].to_csv(reports / "test_predictions.csv", index=False)
    model_importance(selected_model, features).to_csv(reports / "feature_importance.csv", index=False)
    pd.DataFrame([{
        "snapshot_rows": len(dataset),
        "features": len(features),
        "first_target_time": dataset["target_time"].min(),
        "last_target_time": dataset["target_time"].max(),
        "median_horizon_min": dataset["forecast_horizon_min"].median(),
        "median_target_temp": dataset["target_temp"].median(),
    }]).to_csv(reports / "snapshot_summary.csv", index=False)
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    joblib.dump({
        "model": selected_model,
        "features": features,
        "conformal_radius_c": radius,
        "coverage_target": coverage_target,
    }, artifacts / "eaf_temperature_model.joblib")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
