from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

SIGNAL_SUFFIXES = (
    "ActualPosition",
    "ActualVelocity",
    "ActualAcceleration",
    "CommandPosition",
    "CommandVelocity",
    "CommandAcceleration",
    "CurrentFeedback",
    "DCBusVoltage",
    "OutputCurrent",
    "OutputVoltage",
    "OutputPower",
    "SystemInertia",
)
AXIS_PREFIXES = ("X1_", "Y1_", "Z1_", "S1_")


def add_tracking_errors(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for prefix in AXIS_PREFIXES:
        for quantity in ("Position", "Velocity", "Acceleration"):
            actual = f"{prefix}Actual{quantity}"
            command = f"{prefix}Command{quantity}"
            if actual in enriched.columns and command in enriched.columns:
                enriched[f"{prefix}{quantity}TrackingError"] = (
                    pd.to_numeric(enriched[actual], errors="coerce")
                    - pd.to_numeric(enriched[command], errors="coerce")
                )
    return enriched


def select_signal_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column == "M1_CURRENT_FEEDRATE":
            columns.append(column)
            continue
        if column.endswith("TrackingError"):
            columns.append(column)
            continue
        if column.startswith(AXIS_PREFIXES) and column.endswith(SIGNAL_SUFFIXES):
            columns.append(column)
    return [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]


def _slope(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float)
    x -= x.mean()
    y = values - values.mean()
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def summarize_signal(values: pd.Series, statistics: Sequence[str]) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {name: math.nan for name in statistics}

    result: dict[str, float] = {}
    for statistic in statistics:
        if statistic == "mean":
            result[statistic] = float(np.mean(finite))
        elif statistic == "std":
            result[statistic] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        elif statistic == "median":
            result[statistic] = float(np.median(finite))
        elif statistic == "min":
            result[statistic] = float(np.min(finite))
        elif statistic == "max":
            result[statistic] = float(np.max(finite))
        elif statistic == "q25":
            result[statistic] = float(np.quantile(finite, 0.25))
        elif statistic == "q75":
            result[statistic] = float(np.quantile(finite, 0.75))
        elif statistic == "rms":
            result[statistic] = float(np.sqrt(np.mean(np.square(finite))))
        elif statistic == "slope":
            result[statistic] = _slope(finite)
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
    return result


def extract_window_features(
    window: pd.DataFrame,
    *,
    statistics: Sequence[str],
    sample_period_seconds: float,
) -> dict[str, object]:
    if window.empty:
        raise ValueError("Cannot extract features from an empty window.")
    enriched = add_tracking_errors(window)
    signal_columns = select_signal_columns(enriched)
    features: dict[str, object] = {}
    for column in signal_columns:
        summary = summarize_signal(enriched[column], statistics)
        for statistic, value in summary.items():
            features[f"{column}__{statistic}"] = value

    if "Machining_Process" in enriched.columns:
        process = enriched["Machining_Process"].astype(str)
        mode = process.mode(dropna=True)
        features["machining_process"] = mode.iloc[0] if not mode.empty else "unknown"
        features["process_transition_count"] = float(process.ne(process.shift()).sum() - 1)
    else:
        features["machining_process"] = "unknown"
        features["process_transition_count"] = 0.0

    features["known_artifact_fraction"] = (
        float(enriched["_known_artifact"].mean()) if "_known_artifact" in enriched.columns else 0.0
    )
    features["window_duration_seconds"] = float(len(enriched) * sample_period_seconds)
    return features


def make_windows(
    experiment: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    minimum_rows_per_window: int,
    statistics: Sequence[str],
    sample_period_seconds: float,
    target: str,
) -> pd.DataFrame:
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive.")
    if len(experiment) < minimum_rows_per_window:
        return pd.DataFrame()
    if "experiment_id" not in experiment.columns:
        raise ValueError("experiment_id is required before window extraction.")
    if target not in experiment.columns:
        raise ValueError(f"Target {target!r} is not available in the experiment frame.")

    rows: list[dict[str, object]] = []
    starts = list(range(0, max(1, len(experiment) - window_size + 1), step_size))
    if not starts:
        starts = [0]
    tail_start = max(0, len(experiment) - window_size)
    if tail_start > starts[-1]:
        starts.append(tail_start)

    experiment_id = int(experiment["experiment_id"].iloc[0])
    target_value = experiment[target].iloc[0]
    for start in starts:
        stop = min(start + window_size, len(experiment))
        window = experiment.iloc[start:stop]
        if len(window) < minimum_rows_per_window:
            continue
        features = extract_window_features(
            window,
            statistics=statistics,
            sample_period_seconds=sample_period_seconds,
        )
        features.update(
            {
                "experiment_id": experiment_id,
                "window_start": int(start),
                "window_end": int(stop),
                "target": target_value,
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def build_feature_table(
    experiments: Sequence[pd.DataFrame],
    *,
    target: str,
    window_size: int,
    step_size: int,
    minimum_rows_per_window: int,
    statistics: Sequence[str],
    sample_period_seconds: float,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for experiment in experiments:
        target_value = experiment[target].iloc[0] if target in experiment.columns else np.nan
        if pd.isna(target_value) or str(target_value).strip() == "":
            continue
        table = make_windows(
            experiment,
            window_size=window_size,
            step_size=step_size,
            minimum_rows_per_window=minimum_rows_per_window,
            statistics=statistics,
            sample_period_seconds=sample_period_seconds,
            target=target,
        )
        if not table.empty:
            tables.append(table)
    if not tables:
        raise ValueError(f"No feature windows could be created for target {target!r}.")
    return pd.concat(tables, ignore_index=True, sort=False).replace([np.inf, -np.inf], np.nan)
