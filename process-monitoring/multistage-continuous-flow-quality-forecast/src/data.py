from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import output_columns, process_feature_columns


@dataclass
class SupervisedData:
    X: pd.DataFrame
    y: pd.DataFrame
    persistence: pd.DataFrame
    setpoint: pd.DataFrame
    target_setpoint: pd.DataFrame
    feature_time: pd.Series
    target_time: pd.Series
    valid_target: pd.Series
    feature_columns: list[str]
    target_columns: list[str]


def load_source(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "time_stamp" not in frame.columns:
        raise KeyError("Expected time_stamp column")
    frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], errors="coerce")
    if frame["time_stamp"].isna().any():
        raise ValueError("time_stamp contains unparsable values")
    return frame.sort_values("time_stamp").reset_index(drop=True)


def build_supervised(frame: pd.DataFrame, horizon_rows: int) -> SupervisedData:
    if horizon_rows <= 0:
        raise ValueError("horizon_rows must be positive")
    if horizon_rows >= len(frame):
        raise ValueError("horizon_rows must be smaller than dataset length")

    target_cols = output_columns(frame.columns, stage=2, kind="actual")
    setpoint_cols = output_columns(frame.columns, stage=2, kind="setpoint")
    if len(target_cols) != 15 or len(setpoint_cols) != 15:
        raise ValueError("Expected 15 Stage-2 actual and 15 Stage-2 setpoint columns")

    feature_cols = process_feature_columns(frame)
    feature_idx = np.arange(0, len(frame) - horizon_rows)
    target_idx = feature_idx + horizon_rows

    X = frame.iloc[feature_idx][feature_cols].reset_index(drop=True)
    y = frame.iloc[target_idx][target_cols].reset_index(drop=True)
    persistence = frame.iloc[feature_idx][target_cols].reset_index(drop=True)
    setpoint = frame.iloc[feature_idx][setpoint_cols].reset_index(drop=True)
    target_setpoint = frame.iloc[target_idx][setpoint_cols].reset_index(drop=True)
    feature_time = frame.iloc[feature_idx]["time_stamp"].reset_index(drop=True)
    target_time = frame.iloc[target_idx]["time_stamp"].reset_index(drop=True)

    valid_target = target_setpoint.gt(0).all(axis=1) & np.isfinite(y.to_numpy(dtype=float)).all(axis=1)

    return SupervisedData(
        X=X,
        y=y,
        persistence=persistence,
        setpoint=setpoint,
        target_setpoint=target_setpoint,
        feature_time=feature_time,
        target_time=target_time,
        valid_target=pd.Series(valid_target, index=X.index),
        feature_columns=feature_cols,
        target_columns=target_cols,
    )


def chronological_blocks(
    n_rows: int,
    fit_fraction: float,
    selection_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
    purge_rows: int,
) -> dict[str, np.ndarray]:
    fractions = np.array([fit_fraction, selection_fraction, calibration_fraction, test_fraction], dtype=float)
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1")
    if n_rows < 100:
        raise ValueError("Too few rows for chronological split")
    if purge_rows < 0:
        raise ValueError("purge_rows must be non-negative")

    b1 = int(n_rows * fit_fraction)
    b2 = int(n_rows * (fit_fraction + selection_fraction))
    b3 = int(n_rows * (fit_fraction + selection_fraction + calibration_fraction))

    blocks = {
        "fit": np.arange(0, max(0, b1 - purge_rows)),
        "selection": np.arange(min(n_rows, b1 + purge_rows), max(min(n_rows, b2 - purge_rows), min(n_rows, b1 + purge_rows))),
        "calibration": np.arange(min(n_rows, b2 + purge_rows), max(min(n_rows, b3 - purge_rows), min(n_rows, b2 + purge_rows))),
        "test": np.arange(min(n_rows, b3 + purge_rows), n_rows),
    }
    if any(len(v) == 0 for v in blocks.values()):
        raise ValueError("Purge interval leaves an empty split")
    return blocks


def filtered_indices(indices: np.ndarray, valid_mask: pd.Series) -> np.ndarray:
    valid = valid_mask.to_numpy(dtype=bool)
    return indices[valid[indices]]
