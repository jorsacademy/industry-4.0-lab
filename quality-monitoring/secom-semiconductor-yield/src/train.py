from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.base import clone

from .data import block_summary, chronological_blocks, feature_columns, load_secom
from .features import drift_report, fit_feature_screen, panel_features
from .modeling import build_models, choose_compact_candidate, choose_threshold, evaluate_candidate, threshold_metrics


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological SECOM yield benchmark.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(_resolve(project_root, args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    raw_dir = _resolve(project_root, data_cfg["raw_dir"])
    df = load_secom(raw_dir / data_cfg["data_file"], raw_dir / data_cfg["labels_file"])
    features = feature_columns(df)
    split_cfg = cfg["split"]
    blocks = chronological_blocks(df, fit_fraction=float(split_cfg["fit_fraction"]), selection_fraction=float(split_cfg["selection_fraction"]), calibration_fraction=float(split_cfg["calibration_fraction"]), test_fraction=float(split_cfg["test_fraction"]))
    quality_cfg = cfg["quality"]
    screen = fit_feature_screen(blocks.fit, features, max_missing_rate=float(quality_cfg.get("max_missing_rate", 0.50)))
    model_cfg = cfg["model"]
    models = build_models(int(model_cfg.get("random_state", 42)))
    candidate_results, rows = [], []
    for panel in cfg["selection"]["panel_sizes"]:
        cols = panel_features(screen, panel)
        for model_name, model in models.items():
            result = evaluate_candidate(model, blocks.fit, blocks.selection, cols, model_name)
            candidate_results.append(result)
            rows.append({"model": model_name, "panel_request": panel, "panel_size": len(cols), **result.metrics})
    selected = choose_compact_candidate(candidate_results, average_precision_tolerance=float(cfg["selection"].get("average_precision_tolerance", 0.02)))
    development = pd.concat([blocks.fit, blocks.selection], ignore_index=True)
    final_screen = fit_feature_screen(development, features, max_missing_rate=float(quality_cfg.get("max_missing_rate", 0.50)))
    final_features = panel_features(final_screen, selected.panel_size)
    final_model = clone(models[selected.model_name])
    final_model.fit(development[final_features], development["is_fail"])
    calibration_prob = final_model.predict_proba(blocks.calibration[final_features])[:, 1]
    threshold = choose_threshold(blocks.calibration["is_fail"], calibration_prob, min_recall=float(model_cfg.get("min_calibration_recall", 0.70)))
    test_prob = final_model.predict_proba(blocks.test[final_features])[:, 1]
    test_metrics = threshold_metrics(blocks.test["is_fail"], test_prob, threshold)
    calibration_metrics = threshold_metrics(blocks.calibration["is_fail"], calibration_prob, threshold)
    reports, artifacts = project_root / "reports", project_root / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["average_precision", "panel_size"], ascending=[False, True]).to_csv(reports / "panel_model_comparison.csv", index=False)
    selected_table = final_screen.ranking.loc[final_screen.ranking["feature"].isin(final_features)].sort_values("rank")
    selected_table.to_csv(reports / "selected_process_variables.csv", index=False)
    drift_report(development, blocks.test, final_features).to_csv(reports / "drift_report.csv", index=False)
    pred = blocks.test[["source_row", "timestamp", "raw_label", "is_fail"]].copy()
    pred["failure_probability"] = test_prob
    pred["predicted_fail"] = pred["failure_probability"].ge(threshold).astype(int)
    pred.to_csv(reports / "test_predictions.csv", index=False)
    metrics = {
        "source_rows": int(len(df)), "raw_process_variables": int(len(features)),
        "eligible_variables_fit_only": int(len(screen.eligible)), "selected_model": selected.model_name,
        "selected_panel_size": int(len(final_features)), "selection_average_precision": float(selected.metrics["average_precision"]),
        "selection_tolerance": float(cfg["selection"].get("average_precision_tolerance", 0.02)), "operating_threshold": float(threshold),
        "blocks": block_summary(blocks), "calibration_metrics": calibration_metrics, "future_test_metrics": test_metrics,
        "interpretation_note": "Selected anonymous process variables are diagnostic priorities, not identified physical root causes.",
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"pipeline": final_model, "features": final_features, "threshold": threshold, "screen_ranking": final_screen.ranking}, artifacts / "secom_yield_model.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
