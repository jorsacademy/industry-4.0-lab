from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeBlocks:
    fit: pd.DataFrame
    selection: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def load_secom(data_path: str | Path, labels_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    labels_path = Path(labels_path)
    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"])
    X.columns = [f"V{i:03d}" for i in range(X.shape[1])]
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["raw_label", "date", "time"],
        dtype={"raw_label": int, "date": str, "time": str},
    )
    if len(X) != len(labels):
        raise ValueError(f"Feature/label row mismatch: {len(X)} vs {len(labels)}")
    timestamp = pd.to_datetime(
        labels["date"].astype(str) + " " + labels["time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    out = X.copy()
    out["timestamp"] = timestamp
    out["raw_label"] = labels["raw_label"].astype(int).to_numpy()
    out["is_fail"] = out["raw_label"].eq(1).astype(int)
    out["source_row"] = np.arange(len(out), dtype=int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("V")]


def chronological_blocks(
    df: pd.DataFrame,
    *,
    fit_fraction: float,
    selection_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> TimeBlocks:
    fractions = np.array([fit_fraction, selection_fraction, calibration_fraction, test_fraction], dtype=float)
    if (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.")
    work = df.sort_values(["timestamp", "source_row"]).reset_index(drop=True)
    unique_times = pd.Index(work["timestamp"].drop_duplicates().sort_values())
    if len(unique_times) < 8:
        raise ValueError("Too few unique timestamps for four chronological blocks.")
    n = len(unique_times)
    c1 = max(1, min(n - 3, int(np.floor(n * fractions[0]))))
    c2 = max(c1 + 1, min(n - 2, int(np.floor(n * (fractions[0] + fractions[1])))))
    c3 = max(c2 + 1, min(n - 1, int(np.floor(n * (fractions[0] + fractions[1] + fractions[2])))))
    t1, t2, t3 = unique_times[c1], unique_times[c2], unique_times[c3]
    fit = work.loc[work["timestamp"] < t1].copy()
    selection = work.loc[(work["timestamp"] >= t1) & (work["timestamp"] < t2)].copy()
    calibration = work.loc[(work["timestamp"] >= t2) & (work["timestamp"] < t3)].copy()
    test = work.loc[work["timestamp"] >= t3].copy()
    blocks = TimeBlocks(fit, selection, calibration, test)
    for name, block in blocks.__dict__.items():
        if block.empty:
            raise ValueError(f"Chronological block '{name}' is empty.")
        if block["is_fail"].nunique() < 2:
            raise ValueError(f"Chronological block '{name}' does not contain both classes.")
    return blocks


def block_summary(blocks: TimeBlocks) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, block in blocks.__dict__.items():
        result[name] = {
            "rows": int(len(block)),
            "failures": int(block["is_fail"].sum()),
            "failure_rate": float(block["is_fail"].mean()),
            "start": block["timestamp"].min().isoformat(),
            "end": block["timestamp"].max().isoformat(),
        }
    return result
