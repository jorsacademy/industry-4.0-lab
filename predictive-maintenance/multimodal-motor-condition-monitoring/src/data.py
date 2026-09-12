from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ACC_RE = re.compile(r"^[xyz]acc\d+hz$", re.IGNORECASE)
SND_RE = re.compile(r"^snd\d+hz$", re.IGNORECASE)
ID_TO_LABEL = {1: "off", 2: "on", 3: "cap", 4: "out", 5: "unb", 6: "c25", 7: "c75", 8: "vnt"}


@dataclass(frozen=True)
class FrequencyTable:
    frame: pd.DataFrame
    source_path: Path
    acceleration_columns: list[str]
    audio_columns: list[str]
    timestamp_column: str | None


def spectral_columns(columns: list[str] | pd.Index) -> tuple[list[str], list[str]]:
    accel = [str(c) for c in columns if ACC_RE.match(str(c).strip())]
    audio = [str(c) for c in columns if SND_RE.match(str(c).strip())]
    return accel, audio


def _read_header(path: Path) -> list[str]:
    try:
        return [str(c).strip() for c in pd.read_csv(path, nrows=2).columns]
    except Exception:
        return []


def find_frequency_table(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    candidates: list[tuple[int, Path]] = []
    for path in raw_dir.rglob("*.csv"):
        cols = _read_header(path)
        accel, audio = spectral_columns(cols)
        score = len(accel) + len(audio)
        if score:
            candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError("No CSV containing AI4I frequency-feature columns was found.")
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1]


def _resolve_label_column(df: pd.DataFrame) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in ("label", "condition", "state", "class"):
        if name in lowered:
            return lowered[name]
    for col in df.columns:
        series = df[col]
        if series.dtype == object and 2 <= series.nunique(dropna=True) <= 20:
            return str(col)
    return None


def _resolve_id_column(df: pd.DataFrame) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in ("id", "conditionid", "condition_id"):
        if name in lowered:
            return lowered[name]
    return None


def _resolve_timestamp_column(df: pd.DataFrame) -> str | None:
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in ("timestamp", "time", "start", "windowstart", "window_start"):
        if name in lowered:
            return lowered[name]
    return None


def load_frequency_table(raw_dir: str | Path) -> FrequencyTable:
    path = find_frequency_table(raw_dir)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    accel, audio = spectral_columns(df.columns)
    if not accel or not audio:
        raise ValueError("Expected both accelerometer and airborne-sound frequency features.")

    label_col = _resolve_label_column(df)
    id_col = _resolve_id_column(df)
    if label_col is not None:
        condition = df[label_col].astype(str).str.strip().str.lower()
    elif id_col is not None:
        ids = pd.to_numeric(df[id_col], errors="coerce")
        condition = ids.map(ID_TO_LABEL)
    else:
        raise ValueError("Could not resolve condition label or ID column.")
    if condition.isna().any():
        raise ValueError("Condition labels contain unmapped/missing values.")

    timestamp_col = _resolve_timestamp_column(df)
    out = df.copy()
    out["condition"] = condition.to_numpy()
    out["source_row"] = np.arange(len(out), dtype=int)
    if timestamp_col is not None:
        parsed = pd.to_numeric(out[timestamp_col], errors="coerce")
        if parsed.notna().all():
            out["window_time"] = parsed.astype(float)
        else:
            out["window_time"] = pd.to_datetime(out[timestamp_col], errors="coerce")
    else:
        out["window_time"] = np.nan

    sort_cols = ["condition"]
    if timestamp_col is not None:
        sort_cols.append("window_time")
    sort_cols.append("source_row")
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out["window_index"] = out.groupby("condition", sort=False).cumcount()

    for col in accel + audio:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[accel + audio].isna().any().any():
        raise ValueError("Frequency feature matrix contains non-numeric or missing values.")

    return FrequencyTable(out, path, accel, audio, timestamp_col)


def validate_frequency_table(table: FrequencyTable) -> dict[str, object]:
    df = table.frame
    counts = df["condition"].value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "conditions": int(df["condition"].nunique()),
        "condition_counts": {str(k): int(v) for k, v in counts.items()},
        "acceleration_features": int(len(table.acceleration_columns)),
        "audio_features": int(len(table.audio_columns)),
        "frequency_features": int(len(table.acceleration_columns) + len(table.audio_columns)),
        "source_file": str(table.source_path),
        "timestamp_column": table.timestamp_column,
    }


def blocked_purged_folds(df: pd.DataFrame, *, n_splits: int = 5, purge_windows: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if purge_windows < 0:
        raise ValueError("purge_windows must be non-negative")

    condition_positions: dict[str, np.ndarray] = {}
    for condition, group in df.groupby("condition", sort=True):
        ordered = group.sort_values("window_index")
        condition_positions[str(condition)] = ordered.index.to_numpy()
        if len(ordered) < n_splits * 2:
            raise ValueError(f"Condition {condition!r} has too few windows for blocked CV")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for indices in condition_positions.values():
            segments = np.array_split(np.arange(len(indices)), n_splits)
            test_pos = segments[fold_id]
            if len(test_pos) == 0:
                continue
            lo = max(0, int(test_pos.min()) - purge_windows)
            hi = min(len(indices) - 1, int(test_pos.max()) + purge_windows)
            train_mask = np.ones(len(indices), dtype=bool)
            train_mask[lo : hi + 1] = False
            train_parts.append(indices[train_mask])
            test_parts.append(indices[test_pos])
        folds.append((np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))))
    return folds
