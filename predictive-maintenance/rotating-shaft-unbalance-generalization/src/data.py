from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .features import FeatureConfig, extract_window_features

REQUIRED_COLUMNS = ["V_in", "Measured_RPM", "Vibration_1", "Vibration_2", "Vibration_3"]
EXPECTED_RECORDINGS = [f"{severity}{session}.csv" for severity in range(5) for session in ("D", "E")]


def find_member(zf: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in zf.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one archive member named {basename}, found {matches}")
    return matches[0]


def inspect_archive(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        members: dict[str, str] = {}
        for basename in EXPECTED_RECORDINGS:
            member = find_member(zf, basename)
            members[basename] = member
            with zf.open(member) as handle:
                header = pd.read_csv(handle, nrows=0)
            missing = set(REQUIRED_COLUMNS).difference(header.columns)
            if missing:
                raise ValueError(f"{basename}: missing required columns {sorted(missing)}")
    return {
        "archive": str(path),
        "recordings": len(members),
        "members": members,
        "required_columns": REQUIRED_COLUMNS,
    }


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, source_url: str, archive_path: str | Path) -> dict[str, object]:
    archive_path = Path(archive_path)
    manifest = {
        "source_url": source_url,
        "archive_name": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _coerce_chunk(chunk: pd.DataFrame) -> np.ndarray:
    numeric = chunk[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.interpolate(limit_direction="both").fillna(0.0)
    return numeric.to_numpy(dtype=np.float64, copy=False)


def iter_nonoverlap_windows(
    chunks: Iterator[pd.DataFrame],
    *,
    window_samples: int,
    hop_samples: int,
    skip_samples: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield fixed windows from a chunked table without materializing a full recording."""
    if hop_samples < window_samples:
        raise ValueError("hop_samples must be >= window_samples for this leakage-safe benchmark")

    buffer = np.empty((0, len(REQUIRED_COLUMNS)), dtype=np.float64)
    buffer_start = 0
    stream_end = 0
    next_start = int(skip_samples)

    for chunk in chunks:
        values = _coerce_chunk(chunk)
        if buffer.size:
            values = np.vstack([buffer, values])
            base = buffer_start
        else:
            base = stream_end
        end = base + len(values)

        while next_start + window_samples <= end:
            local = next_start - base
            if local < 0:
                raise RuntimeError("window stream bookkeeping error")
            yield next_start, values[local : local + window_samples]
            next_start += hop_samples

        keep_from = max(0, next_start - base)
        if keep_from < len(values):
            buffer = values[keep_from:].copy()
            buffer_start = base + keep_from
        else:
            buffer = np.empty((0, len(REQUIRED_COLUMNS)), dtype=np.float64)
            buffer_start = end
        stream_end = end


def recording_feature_table(
    archive_path: str | Path,
    recording: str,
    *,
    signal_cfg: dict[str, float | int],
) -> pd.DataFrame:
    severity = int(recording[0])
    session = recording[1]
    sample_rate = int(signal_cfg["sample_rate_hz"])
    window_samples = int(round(sample_rate * float(signal_cfg["window_seconds"])))
    hop_samples = int(round(sample_rate * float(signal_cfg["hop_seconds"])))
    skip_samples = int(round(sample_rate * float(signal_cfg.get("skip_seconds", 0.0))))
    chunk_rows = int(signal_cfg.get("chunk_rows", 262144))
    feature_cfg = FeatureConfig(
        sample_rate_hz=float(sample_rate),
        order_min=float(signal_cfg["order_min"]),
        order_max=float(signal_cfg["order_max"]),
        order_step=float(signal_cfg["order_step"]),
    )

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as zf:
        member = find_member(zf, f"{recording}.csv")
        with zf.open(member) as handle:
            chunks = pd.read_csv(
                handle,
                usecols=REQUIRED_COLUMNS,
                chunksize=chunk_rows,
                low_memory=False,
            )
            for window_index, (start_sample, window) in enumerate(
                iter_nonoverlap_windows(
                    chunks,
                    window_samples=window_samples,
                    hop_samples=hop_samples,
                    skip_samples=skip_samples,
                )
            ):
                record: dict[str, object] = {
                    "recording": recording,
                    "session": session,
                    "severity": severity,
                    "window_index": window_index,
                    "start_sample": int(start_sample),
                    "start_sec": float(start_sample / sample_rate),
                }
                record.update(extract_window_features(window, REQUIRED_COLUMNS, feature_cfg))
                rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No windows extracted from {recording}")
    return frame


def build_feature_table(
    archive_path: str | Path,
    signal_cfg: dict[str, float | int],
    n_folds: int = 5,
) -> pd.DataFrame:
    frames = [recording_feature_table(archive_path, f"{s}{session}", signal_cfg=signal_cfg)
              for session in ("D", "E") for s in range(5)]
    frame = pd.concat(frames, ignore_index=True)
    frame["dev_block"] = -1
    for recording, idx in frame.groupby("recording", sort=False).groups.items():
        positions = np.arange(len(idx))
        blocks = np.minimum((positions * n_folds) // max(len(idx), 1), n_folds - 1)
        frame.loc[list(idx), "dev_block"] = blocks.astype(int)
    return frame


def compact_feature_columns(frame: pd.DataFrame, sensors: tuple[int, ...] = (1, 2, 3)) -> list[str]:
    base = [c for c in frame.columns if c.startswith("compact__") and "__s" not in c]
    sensor_cols = [
        c for c in frame.columns
        if c.startswith("compact__") and any(f"__s{s}__" in c for s in sensors)
    ]
    return sorted(base + sensor_cols)


def order_feature_columns(frame: pd.DataFrame, sensors: tuple[int, ...] = (1, 2, 3)) -> list[str]:
    base = [c for c in frame.columns if c.startswith("order__") and "__s" not in c]
    sensor_cols = [
        c for c in frame.columns
        if c.startswith("order__") and any(f"__s{s}__" in c for s in sensors)
    ]
    return sorted(base + sensor_cols)
