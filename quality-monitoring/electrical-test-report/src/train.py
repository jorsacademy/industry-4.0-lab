from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from .data import feature_target_group, load_electrical_report
from .modeling import build_candidates, choose_threshold_for_recall, evaluate_candidates, feature_importance_table
from .monitoring import summarize_lots


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lot-safe electrical-test defect models.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    monitor_cfg = cfg["monitoring"]

    df = load_electrical_report(
        data_cfg["path"],
        target_column=data_cfg.get("target_column", "Result"),
        group_column=data_cfg.get("group_column", "LOT"),
        time_column=data_cfg.get("time_column", "Time"),
        pass_code=int(data_cfg.get("pass_code", 1)),
    )
    X, y, groups, features = feature_target_group(df, pass_code=int(data_cfg.get("pass_code", 1)))

    results = evaluate_candidates(X, y, groups, n_splits=int(model_cfg.get("cv_splits", 5)), random_state=int(model_cfg.get("random_state", 42)))
    primary = model_cfg.get("primary_metric", "average_precision")
    best = max(results, key=lambda r: r.metrics[primary])
    threshold = choose_threshold_for_recall(y, best.probabilities, float(model_cfg.get("target_recall", 0.90)))

    reports = Path("reports")
    artifacts = Path("artifacts")
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)

    metrics = {
        "primary_metric": primary,
        "selected_model": best.model_name,
        "operating_threshold": threshold,
        "candidates": {r.model_name: r.metrics for r in results},
        "records": int(len(df)),
        "defects": int(y.sum()),
        "lots": int(pd.Series(groups).nunique()),
        "measurement_features": features,
    }
    (reports / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    oof = df[["Time", "LOT", "Result", "timestamp", "source_row"]].copy()
    oof["actual_defect"] = y.to_numpy()
    oof["defect_probability"] = best.probabilities
    oof["alert"] = oof["defect_probability"].ge(threshold)
    oof.to_csv(reports / "oof_predictions.csv", index=False)

    lot_report = summarize_lots(df, min_lot_size=int(monitor_cfg.get("stable_lot_min_size", 100)), warning_rate=float(monitor_cfg.get("warning_defect_rate", 0.05)))
    lot_scores = oof.groupby("LOT").agg(model_risk=("defect_probability", "mean"), model_alert_share=("alert", "mean")).reset_index()
    lot_report.merge(lot_scores, on="LOT", how="left").to_csv(reports / "lot_quality_summary.csv", index=False)

    model = build_candidates(int(model_cfg.get("random_state", 42)))[best.model_name]
    model.fit(X, y)
    joblib.dump({"pipeline": model, "features": features, "threshold": threshold}, artifacts / "defect_model.joblib")
    feature_importance_table(model, features).to_csv(reports / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
