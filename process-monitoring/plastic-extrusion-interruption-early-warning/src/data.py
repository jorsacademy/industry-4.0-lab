from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_COL = "Datum"
THICKNESS_COLS = [f"ST110_VAREx_{i}_SDickeIst" for i in range(4)]
OUTPUT_COLS = [f"ST110_VAREx_{i}_GesamtDS" for i in range(4)]
WINDER_COLS = ["ST113_VARLmpRun", "ST114_VARLmpRun"]
HAUL_ON_COL = "ST112_VARAbzug_1_IstEin"
HAUL_CLOSED_COL = "ST112_VARAbzug_1_IstZu"
HAUL_SPEED_COL = "ST112_VARAbzug_1_SollSpeed"
ACTIVE_CONTROL_COLS = [HAUL_ON_COL, HAUL_CLOSED_COL, HAUL_SPEED_COL, *WINDER_COLS]
DIRECT_STATE_COLS = [*THICKNESS_COLS, *OUTPUT_COLS, *ACTIVE_CONTROL_COLS]


@dataclass
class PreparedData:
    frame: pd.DataFrame
    timestamps: pd.Series
    segments: pd.Series
    active: pd.Series
    event_onset: pd.Series
    target: pd.Series
    lead_minutes: pd.Series
    predictor_columns: list[str]


def locate_source_csv(raw_dir: str | Path) -> Path:
    raw = Path(raw_dir)
    candidates = [p for p in raw.rglob("*.csv") if p.name != "stat.csv"]
    if not candidates:
        raise FileNotFoundError(f"No production CSV found under {raw}")
    return max(candidates, key=lambda p: p.stat().st_size)


def load_source(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = [
        c
        for c in [
            TIMESTAMP_COL,
            *THICKNESS_COLS,
            *OUTPUT_COLS,
            *ACTIVE_CONTROL_COLS,
        ]
        if c not in frame.columns
    ]
    if missing:
        raise KeyError(f"Missing required source columns: {missing}")

    for col in frame.columns:
        if col == TIMESTAMP_COL:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def parse_timestamps(frame: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(frame[TIMESTAMP_COL], format="%d.%m.%Y %H:%M", errors="coerce")
    if ts.isna().any():
        raise ValueError("Source contains unparsable timestamps")
    if not ts.is_monotonic_increasing:
        raise ValueError("Source timestamps are not monotonic")
    return ts


def segment_ids(timestamps: pd.Series, segment_gap_minutes: float) -> pd.Series:
    gap_seconds = timestamps.diff().dt.total_seconds()
    starts = gap_seconds.gt(float(segment_gap_minutes) * 60.0).fillna(False)
    return starts.cumsum().astype("int32")


def active_production_mask(frame: pd.DataFrame, minimum_haul_speed: float = 10.0) -> pd.Series:
    positive_thickness = frame[THICKNESS_COLS].gt(0).all(axis=1)
    positive_output = frame[OUTPUT_COLS].gt(0).all(axis=1)
    haul_ready = frame[HAUL_ON_COL].eq(1) & frame[HAUL_CLOSED_COL].eq(1)
    speed_ready = frame[HAUL_SPEED_COL].ge(float(minimum_haul_speed))
    any_winder = frame[WINDER_COLS].eq(1).any(axis=1)
    return (positive_thickness & positive_output & haul_ready & speed_ready & any_winder).astype(bool)


def interruption_onsets(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    active: pd.Series,
    onset_max_gap_minutes: float = 3.0,
) -> pd.Series:
    all_zero = frame[THICKNESS_COLS].eq(0).all(axis=1)
    transition = all_zero & ~all_zero.shift(fill_value=False)
    normal_gap = timestamps.diff().dt.total_seconds().le(float(onset_max_gap_minutes) * 60.0)
    return (transition & active.shift(fill_value=False) & normal_gap.fillna(False)).astype(bool)


def upcoming_event_target(
    timestamps: pd.Series,
    segments: pd.Series,
    active: pd.Series,
    event_onset: pd.Series,
    warning_horizon_minutes: float,
) -> tuple[pd.Series, pd.Series]:
    event_time = timestamps.where(event_onset)
    helper = pd.DataFrame({"segment": segments, "event_time": event_time})
    next_event = helper.groupby("segment", sort=False)["event_time"].bfill()
    delta_min = (next_event - timestamps).dt.total_seconds() / 60.0
    target = active & delta_min.gt(0) & delta_min.le(float(warning_horizon_minutes))
    lead = delta_min.where(target)
    return target.astype("int8"), lead


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {TIMESTAMP_COL, *DIRECT_STATE_COLS}
    columns: list[str] = []
    for col in frame.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            columns.append(col)
    return columns


def prepare_data(
    path: str | Path,
    warning_horizon_minutes: float = 20.0,
    segment_gap_minutes: float = 5.0,
    onset_max_gap_minutes: float = 3.0,
    minimum_haul_speed: float = 10.0,
) -> PreparedData:
    frame = load_source(path)
    timestamps = parse_timestamps(frame)
    segments = segment_ids(timestamps, segment_gap_minutes)
    active = active_production_mask(frame, minimum_haul_speed)
    event_onset = interruption_onsets(frame, timestamps, active, onset_max_gap_minutes)
    target, lead = upcoming_event_target(
        timestamps,
        segments,
        active,
        event_onset,
        warning_horizon_minutes,
    )
    return PreparedData(
        frame=frame,
        timestamps=timestamps,
        segments=segments,
        active=active,
        event_onset=event_onset,
        target=target,
        lead_minutes=lead,
        predictor_columns=predictor_columns(frame),
    )
