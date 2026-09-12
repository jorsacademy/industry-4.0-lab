from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import locate_source_csv, prepare_data
from .modeling import (
    choose_threshold,
    classification_metrics,
    drift_diagnostics,
    fitted_feature_importance,
    make_model,
    probability,
    rank_features,
    ranking_metrics,
    screen_features,
)

ROOT = Path(__file__).resolve().parents[1]


def chronological_blocks(
    timestamps: pd.Series,
    active: pd.Series,
    fit_fraction: float,
    selection_fraction: float,
    calibration_fraction: float,
    purge_minutes: float,
) -> tuple[dict[str, np.ndarray], dict[str, pd.Timestamp]]:
    n = len(timestamps)
    b1 = int(np.floor(n * fit_fraction))
    b2 = int(np.floor(n * (fit_fraction + selection_fraction)))
    b3 = int(np.floor(n * (fit_fraction + selection_fraction + calibration_fraction)))
    t1, t2, t3 = timestamps.iloc[b1], timestamps.iloc[b2], timestamps.iloc[b3]
    purge = pd.Timedelta(minutes=float(purge_minutes))
    row = np.arange(n)

    masks = {
        "fit": (row < b1) & timestamps.le(t1 - purge).to_numpy() & active.to_numpy(),
        "selection": (
            (row >= b1)
            & (row < b2)
            & timestamps.ge(t1 + purge).to_numpy()
            & timestamps.le(t2 - purge).to_numpy()
            & active.to_numpy()
        ),
        "calibration": (
            (row >= b2)
            & (row < b3)
            & timestamps.ge(t2 + purge).to_numpy()
            & timestamps.le(t3 - purge).to_numpy()
            & active.to_numpy()
        ),
        "test": (row >= b3) & timestamps.ge(t3 + purge).to_numpy() & active.to_numpy(),
    }
    blocks = {name: np.flatnonzero(mask) for name, mask in masks.items()}
    if any(len(rows) == 0 for rows in blocks.values()):
        raise ValueError(f"Empty chronological block: { {k: len(v) for k, v in blocks.items()} }")
    bounds = {
        "selection_boundary": t1,
        "calibration_boundary": t2,
        "test_boundary": t3,
        "test_scoring_start": t3 + purge,
    }
    return blocks, bounds


def block_summary(rows: np.ndarray, timestamps: pd.Series, target: pd.Series) -> dict[str, object]:
    y = target.iloc[rows]
    return {
        "rows": int(len(rows)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "start": timestamps.iloc[rows[0]].isoformat(),
        "end": timestamps.iloc[rows[-1]].isoformat(),
    }


def event_metrics(
    timestamps: pd.Series,
    segments: pd.Series,
    event_onset: pd.Series,
    test_rows: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    test_scoring_start: pd.Timestamp,
    horizon_minutes: float,
) -> dict[str, object]:
    horizon = pd.Timedelta(minutes=float(horizon_minutes))
    eligible_events = np.flatnonzero(
        (
            event_onset
            & timestamps.ge(test_scoring_start + horizon)
        ).to_numpy()
    )
    row_ts = timestamps.iloc[test_rows].to_numpy(dtype="datetime64[ns]")
    row_seg = segments.iloc[test_rows].to_numpy()
    alarms = probabilities >= float(threshold)

    caught = 0
    leads: list[float] = []
    opportunities = 0
    for event_row in eligible_events:
        event_time = timestamps.iloc[event_row]
        event_seg = segments.iloc[event_row]
        low = np.datetime64(event_time - horizon)
        high = np.datetime64(event_time)
        candidate = (row_seg == event_seg) & (row_ts >= low) & (row_ts < high)
        if not candidate.any():
            continue
        opportunities += 1
        alarm_idx = np.flatnonzero(candidate & alarms)
        if len(alarm_idx):
            caught += 1
            first_alarm_time = pd.Timestamp(row_ts[alarm_idx[0]])
            leads.append(float((event_time - first_alarm_time).total_seconds() / 60.0))

    return {
        "eligible_events_with_full_warning_window": int(opportunities),
        "events_warned": int(caught),
        "event_recall": float(caught / opportunities) if opportunities else None,
        "first_warning_lead_minutes_mean": float(np.mean(leads)) if leads else None,
        "first_warning_lead_minutes_median": float(np.median(leads)) if leads else None,
        "first_warning_lead_minutes_p25": float(np.quantile(leads, 0.25)) if leads else None,
        "first_warning_lead_minutes_p75": float(np.quantile(leads, 0.75)) if leads else None,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def run(config_path: str | Path) -> dict[str, object]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw_dir = ROOT / "data" / "raw"
    source_path = locate_source_csv(raw_dir)

    target_cfg = cfg["target"]
    prepared = prepare_data(
        source_path,
        warning_horizon_minutes=target_cfg["warning_horizon_minutes"],
        segment_gap_minutes=target_cfg["segment_gap_minutes"],
        onset_max_gap_minutes=target_cfg["onset_max_gap_minutes"],
        minimum_haul_speed=target_cfg["minimum_haul_speed"],
    )
    expected_rows = int(cfg["source"]["expected_rows"])
    expected_columns = int(cfg["source"]["expected_columns"])
    if len(prepared.frame) != expected_rows or prepared.frame.shape[1] != expected_columns:
        raise ValueError(
            f"Unexpected source shape {prepared.frame.shape}; expected {(expected_rows, expected_columns)}"
        )

    eval_cfg = cfg["evaluation"]
    blocks, bounds = chronological_blocks(
        prepared.timestamps,
        prepared.active,
        fit_fraction=float(eval_cfg["fit_fraction"]),
        selection_fraction=float(eval_cfg["selection_fraction"]),
        calibration_fraction=float(eval_cfg["calibration_fraction"]),
        purge_minutes=float(eval_cfg["purge_minutes"]),
    )
    for name, rows in blocks.items():
        if prepared.target.iloc[rows].sum() < 5:
            raise ValueError(f"Too few positive upcoming-interruption rows in {name} block")

    feature_cfg = cfg["features"]
    eligible = screen_features(
        prepared.frame,
        blocks["fit"],
        prepared.predictor_columns,
        max_missing_rate=float(feature_cfg["max_missing_rate"]),
    )

    model_cfg = cfg["model"]
    ranking = rank_features(
        prepared.frame,
        prepared.target,
        blocks["fit"],
        eligible,
        random_state=int(model_cfg["random_state"]),
        n_estimators=int(model_cfg["ranking_trees"]),
        sample_rows=int(feature_cfg["ranking_sample_rows"]),
    )
    ranked = ranking["feature"].tolist()
    panel_sizes = sorted(
        set(
            [min(int(k), len(ranked)) for k in feature_cfg["panel_sizes"]]
            + [len(ranked)]
        )
    )

    y_fit = prepared.target.iloc[blocks["fit"]].to_numpy(dtype=np.int8)
    y_selection = prepared.target.iloc[blocks["selection"]].to_numpy(dtype=np.int8)
    candidates: list[dict[str, object]] = []
    fitted_candidates: dict[tuple[str, int], object] = {}

    for panel_size in panel_sizes:
        columns = ranked[:panel_size]
        for model_name in ["logistic", "extra_trees"]:
            model = make_model(
                model_name,
                random_state=int(model_cfg["random_state"]),
                extra_trees=int(model_cfg["extra_trees"]),
                min_samples_leaf=int(model_cfg["min_samples_leaf"]),
            )
            model.fit(prepared.frame.loc[blocks["fit"], columns], y_fit)
            prob = probability(model, prepared.frame, blocks["selection"], columns)
            metrics = ranking_metrics(y_selection, prob)
            record = {
                "model": model_name,
                "panel_size": int(panel_size),
                "fit_rows": int(len(blocks["fit"])),
                "selection_rows": int(len(blocks["selection"])),
                **metrics,
            }
            candidates.append(record)
            fitted_candidates[(model_name, int(panel_size))] = model

    comparison = pd.DataFrame(candidates)
    best_ap = float(comparison["average_precision"].max())
    tolerance = float(feature_cfg["compact_tolerance_ap"])
    compact = comparison[comparison["average_precision"] >= best_ap - tolerance].copy()
    compact = compact.sort_values(
        ["panel_size", "average_precision", "roc_auc", "model"],
        ascending=[True, False, False, True],
    )
    selected_row = compact.iloc[0]
    selected_model_name = str(selected_row["model"])
    selected_panel_size = int(selected_row["panel_size"])
    selected_columns = ranked[:selected_panel_size]

    history_rows = np.sort(np.concatenate([blocks["fit"], blocks["selection"]]))
    y_history = prepared.target.iloc[history_rows].to_numpy(dtype=np.int8)
    final_model = make_model(
        selected_model_name,
        random_state=int(model_cfg["random_state"]),
        extra_trees=int(model_cfg["extra_trees"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
    )
    final_model.fit(prepared.frame.loc[history_rows, selected_columns], y_history)

    y_cal = prepared.target.iloc[blocks["calibration"]].to_numpy(dtype=np.int8)
    p_cal = probability(final_model, prepared.frame, blocks["calibration"], selected_columns)
    threshold = choose_threshold(y_cal, p_cal, float(model_cfg["min_calibration_recall"]))
    calibration_metrics = classification_metrics(y_cal, p_cal, threshold.threshold)

    y_test = prepared.target.iloc[blocks["test"]].to_numpy(dtype=np.int8)
    p_test = probability(final_model, prepared.frame, blocks["test"], selected_columns)
    test_metrics = classification_metrics(y_test, p_test, threshold.threshold)
    event_result = event_metrics(
        prepared.timestamps,
        prepared.segments,
        prepared.event_onset,
        blocks["test"],
        p_test,
        threshold.threshold,
        bounds["test_scoring_start"],
        float(target_cfg["warning_horizon_minutes"]),
    )

    drift = drift_diagnostics(
        prepared.frame,
        blocks["fit"],
        blocks["test"],
        selected_columns,
    )
    importance = fitted_feature_importance(final_model, selected_columns)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    comparison.sort_values(
        ["average_precision", "panel_size"], ascending=[False, True]
    ).to_csv(reports / "model_selection.csv", index=False)
    drift.to_csv(reports / "drift_diagnostics.csv", index=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    split_summary = {
        name: block_summary(rows, prepared.timestamps, prepared.target)
        for name, rows in blocks.items()
    }
    metrics: dict[str, object] = {
        "source": {
            "rows": int(len(prepared.frame)),
            "columns": int(prepared.frame.shape[1]),
            "timestamp_start": prepared.timestamps.iloc[0].isoformat(),
            "timestamp_end": prepared.timestamps.iloc[-1].isoformat(),
            "continuous_segments": int(prepared.segments.nunique()),
            "active_rows_full_sequence": int(prepared.active.sum()),
            "interruption_proxy_onsets_full_sequence": int(prepared.event_onset.sum()),
            "explicit_fault_label_present": False,
        },
        "target_contract": {
            "name": "active_line_interruption_proxy_within_20_minutes",
            "warning_horizon_minutes": float(target_cfg["warning_horizon_minutes"]),
            "segment_gap_minutes": float(target_cfg["segment_gap_minutes"]),
            "onset_max_gap_minutes": float(target_cfg["onset_max_gap_minutes"]),
            "minimum_haul_speed": float(target_cfg["minimum_haul_speed"]),
            "note": "Proxy onset is active production transitioning to all four layer-thickness channels at zero; source has no authoritative stop-cause label.",
        },
        "splits": split_summary,
        "purge_minutes": float(eval_cfg["purge_minutes"]),
        "feature_screen": {
            "raw_numeric_predictors_after_direct_state_exclusion": int(len(prepared.predictor_columns)),
            "eligible_fit_only_predictors": int(len(eligible)),
        },
        "selected": {
            "model": selected_model_name,
            "panel_size": selected_panel_size,
            "selection_average_precision": float(selected_row["average_precision"]),
            "selection_roc_auc": float(selected_row["roc_auc"]),
            "selection_brier": float(selected_row["brier"]),
            "compact_tolerance_ap": tolerance,
        },
        "calibration": {
            "threshold_method": threshold.method,
            **calibration_metrics,
        },
        "future_test": {
            **test_metrics,
            **event_result,
        },
        "interpretation_note": "Observational active-line interruption proxy, not an authoritative film-break/failure label. Final future test is consumed and closed to further selection.",
    }
    metrics = json_ready(metrics)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    source_summary = {
        "rows": int(len(prepared.frame)),
        "columns": int(prepared.frame.shape[1]),
        "timestamps": {
            "start": prepared.timestamps.iloc[0].isoformat(),
            "end": prepared.timestamps.iloc[-1].isoformat(),
            "duplicate_timestamps": int(prepared.timestamps.duplicated().sum()),
        },
        "continuous_segments": int(prepared.segments.nunique()),
        "active_rows": int(prepared.active.sum()),
        "interruption_proxy_onsets": int(prepared.event_onset.sum()),
        "explicit_fault_label_present": False,
    }
    (reports / "source_summary.json").write_text(
        json.dumps(source_summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
