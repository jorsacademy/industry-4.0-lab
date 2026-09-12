from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import build_supervised, chronological_blocks, filtered_indices, load_source
from .modeling import (
    block_bootstrap_improvement_ci,
    conformal_radius,
    drift_table,
    feature_importance_table,
    interval_diagnostics,
    make_model,
    per_target_metrics,
    regression_metrics,
    target_scale,
)

ROOT = Path(__file__).resolve().parents[1]


def _time_range(times: pd.Series, idx: np.ndarray) -> dict[str, object]:
    if len(idx) == 0:
        return {"rows": 0, "start": None, "end": None}
    values = times.iloc[idx]
    return {
        "rows": int(len(idx)),
        "start": values.iloc[0].isoformat(),
        "end": values.iloc[-1].isoformat(),
    }


def _as_numpy(frame: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return frame.iloc[idx].to_numpy(dtype=float)


def main(config_path: str | Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    source_path = ROOT / cfg["source"]["csv"]
    frame = load_source(source_path)

    sample_period = int(cfg["forecast"]["sample_period_seconds"])
    horizons = [int(x) for x in cfg["forecast"]["candidate_horizons_seconds"]]
    purge_rows = int(np.ceil(cfg["split"]["purge_seconds"] / sample_period))
    model_names = ["ridge", "extra_trees"]

    selection_rows: list[dict[str, object]] = []
    prepared: dict[int, tuple] = {}

    for horizon_seconds in horizons:
        if horizon_seconds % sample_period != 0:
            raise ValueError("Candidate horizons must be divisible by sample_period_seconds")
        horizon_rows = horizon_seconds // sample_period
        data = build_supervised(frame, horizon_rows=horizon_rows)
        blocks = chronological_blocks(
            len(data.X),
            fit_fraction=float(cfg["split"]["fit_fraction"]),
            selection_fraction=float(cfg["split"]["selection_fraction"]),
            calibration_fraction=float(cfg["split"]["calibration_fraction"]),
            test_fraction=float(cfg["split"]["test_fraction"]),
            purge_rows=purge_rows,
        )
        blocks = {name: filtered_indices(idx, data.valid_target) for name, idx in blocks.items()}
        if min(len(x) for x in blocks.values()) < 100:
            raise ValueError(f"Too few valid rows after filtering for {horizon_seconds}s horizon")

        fit_idx = blocks["fit"]
        sel_idx = blocks["selection"]
        scale = target_scale(_as_numpy(data.y, fit_idx))
        persistence_sel = regression_metrics(
            _as_numpy(data.y, sel_idx), _as_numpy(data.persistence, sel_idx), scale
        )
        setpoint_sel = regression_metrics(
            _as_numpy(data.y, sel_idx), _as_numpy(data.setpoint, sel_idx), scale
        )

        for model_name in model_names:
            model = make_model(model_name, cfg["models"], seed)
            model.fit(data.X.iloc[fit_idx], _as_numpy(data.y, fit_idx))
            pred = model.predict(data.X.iloc[sel_idx])
            metrics = regression_metrics(_as_numpy(data.y, sel_idx), pred, scale)
            selection_rows.append({
                "horizon_seconds": horizon_seconds,
                "model": model_name,
                "features": len(data.feature_columns),
                "fit_rows": len(fit_idx),
                "selection_rows": len(sel_idx),
                "normalized_mae": metrics.nmae,
                "mean_mae": metrics.mean_mae,
                "mean_rmse": metrics.mean_rmse,
                "mean_r2": metrics.mean_r2,
                "persistence_normalized_mae": persistence_sel.nmae,
                "setpoint_normalized_mae": setpoint_sel.nmae,
            })

        prepared[horizon_seconds] = (data, blocks, scale)

    selection = pd.DataFrame(selection_rows).sort_values(
        ["normalized_mae", "horizon_seconds", "model"], ascending=[True, True, True]
    ).reset_index(drop=True)
    chosen = selection.iloc[0]
    chosen_horizon = int(chosen["horizon_seconds"])
    chosen_model_name = str(chosen["model"])
    data, blocks, fit_scale = prepared[chosen_horizon]

    train_idx = np.concatenate([blocks["fit"], blocks["selection"]])
    cal_idx = blocks["calibration"]
    test_idx = blocks["test"]

    final_scale = target_scale(_as_numpy(data.y, train_idx))
    final_model = make_model(chosen_model_name, cfg["models"], seed)
    final_model.fit(data.X.iloc[train_idx], _as_numpy(data.y, train_idx))

    pred_cal = final_model.predict(data.X.iloc[cal_idx])
    pred_test = final_model.predict(data.X.iloc[test_idx])
    y_test = _as_numpy(data.y, test_idx)
    persistence_test = _as_numpy(data.persistence, test_idx)
    setpoint_test = _as_numpy(data.setpoint, test_idx)

    final_metrics = regression_metrics(y_test, pred_test, final_scale)
    persistence_metrics = regression_metrics(y_test, persistence_test, final_scale)
    setpoint_metrics = regression_metrics(y_test, setpoint_test, final_scale)

    coverage = float(cfg["uncertainty"]["coverage"])
    radius = conformal_radius(_as_numpy(data.y, cal_idx), pred_cal, coverage=coverage)
    interval_report = interval_diagnostics(y_test, pred_test, radius, final_scale)

    target_report = per_target_metrics(
        y_test,
        pred_test,
        persistence_test,
        setpoint_test,
        final_scale,
    ).merge(interval_report, on="target_index", how="left")
    target_report.insert(1, "target", data.target_columns)

    model_row_error = np.mean(np.abs(y_test - pred_test) / final_scale, axis=1)
    persistence_row_error = np.mean(np.abs(y_test - persistence_test) / final_scale, axis=1)
    bootstrap = block_bootstrap_improvement_ci(
        model_error=model_row_error,
        baseline_error=persistence_row_error,
        block_length=int(cfg["bootstrap"]["block_length_rows"]),
        repetitions=int(cfg["bootstrap"]["repetitions"]),
        seed=seed,
    )

    drift = drift_table(data.X.iloc[blocks["fit"]], data.X.iloc[test_idx])
    importance = feature_importance_table(final_model, data.feature_columns)

    reports = ROOT / "reports"
    artifacts = ROOT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    selection.to_csv(reports / "model_selection.csv", index=False)
    target_report.to_csv(reports / "per_target_metrics.csv", index=False)
    interval_report.to_csv(reports / "interval_coverage.csv", index=False)
    drift.head(30).to_csv(reports / "drift_diagnostics.csv", index=False)
    importance.head(50).to_csv(reports / "feature_importance.csv", index=False)

    metrics_payload = {
        "source": {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "sample_period_seconds": sample_period,
            "stage2_targets": len(data.target_columns),
            "production_runs": 1,
        },
        "selection_contract": "forecast horizon and model family selected only on historical fit/selection blocks; calibration used only for conformal residuals; final future test opened once after selection",
        "selected": {
            "horizon_seconds": chosen_horizon,
            "model": chosen_model_name,
            "features": len(data.feature_columns),
            "selection_normalized_mae": float(chosen["normalized_mae"]),
        },
        "splits": {
            "fit": _time_range(data.target_time, blocks["fit"]),
            "selection": _time_range(data.target_time, blocks["selection"]),
            "calibration": _time_range(data.target_time, blocks["calibration"]),
            "test": _time_range(data.target_time, blocks["test"]),
            "purge_seconds": int(cfg["split"]["purge_seconds"]),
        },
        "future_test": {
            "normalized_mae": final_metrics.nmae,
            "mean_mae": final_metrics.mean_mae,
            "mean_rmse": final_metrics.mean_rmse,
            "mean_r2": final_metrics.mean_r2,
        },
        "persistence_baseline": {
            "normalized_mae": persistence_metrics.nmae,
            "mean_mae": persistence_metrics.mean_mae,
            "mean_rmse": persistence_metrics.mean_rmse,
            "mean_r2": persistence_metrics.mean_r2,
        },
        "current_setpoint_baseline": {
            "normalized_mae": setpoint_metrics.nmae,
            "mean_mae": setpoint_metrics.mean_mae,
            "mean_rmse": setpoint_metrics.mean_rmse,
            "mean_r2": setpoint_metrics.mean_r2,
        },
        "model_vs_persistence": {
            "relative_nmae_improvement": float(
                (persistence_metrics.nmae - final_metrics.nmae) / persistence_metrics.nmae
            ) if persistence_metrics.nmae > 0 else None,
            "block_bootstrap": bootstrap,
        },
        "conformal": {
            "target_coverage": coverage,
            "mean_test_coverage": float(interval_report["coverage"].mean()),
            "median_normalized_half_width": float(interval_report["normalized_half_width"].median()),
        },
        "interpretation_note": "Single-run chronological evaluation; the selected horizon is an empirical forecasting alignment, not verified material residence time. Final test is consumed and closed to further selection.",
    }
    (reports / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    joblib.dump(
        {
            "model": final_model,
            "horizon_seconds": chosen_horizon,
            "features": data.feature_columns,
            "targets": data.target_columns,
            "conformal_radius": radius,
            "target_scale": final_scale,
        },
        artifacts / "continuous_flow_quality_model.joblib",
        compress=3,
    )

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
