from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_ALIASES = {
    "machining_completed": "machining_finalized",
    "feed_rate": "feedrate",
}

KNOWN_ARTIFACT_COLUMNS = {
    "M1_CURRENT_FEEDRATE": 50,
    "X1_ActualPosition": 198,
}


def load_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns=TARGET_ALIASES)
    if "No" not in frame.columns:
        raise ValueError("Metadata must contain experiment identifier column 'No'.")
    frame["No"] = pd.to_numeric(frame["No"], errors="raise").astype(int)
    if frame["No"].duplicated().any():
        raise ValueError("Metadata contains duplicate experiment identifiers.")
    return frame.sort_values("No").reset_index(drop=True)


def experiment_id_from_path(path: str | Path) -> int:
    match = re.search(r"experiment_(\d+)\.csv$", Path(path).name)
    if not match:
        raise ValueError(f"Cannot infer experiment id from: {path}")
    return int(match.group(1))


def discover_experiment_files(raw_dir: str | Path, pattern: str = "experiment_*.csv") -> list[Path]:
    files = sorted(Path(raw_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No experiment files matching {pattern!r} under {raw_dir}")
    return files


def load_experiment(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Machining_Process" not in frame.columns:
        raise ValueError(f"Missing Machining_Process in {path}")
    return frame


def known_artifact_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column, value in KNOWN_ARTIFACT_COLUMNS.items():
        if column in frame.columns:
            mask |= pd.to_numeric(frame[column], errors="coerce").eq(value)
    if "M1_CURRENT_PROGRAM_NUMBER" in frame.columns:
        program = pd.to_numeric(frame["M1_CURRENT_PROGRAM_NUMBER"], errors="coerce")
        mask |= program.notna() & program.ne(0)
    return mask


def clean_experiment(frame: pd.DataFrame, *, drop_known_artifacts: bool = True) -> pd.DataFrame:
    cleaned = frame.copy()
    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_columns] = cleaned[numeric_columns].replace([np.inf, -np.inf], np.nan)
    artifact_mask = known_artifact_mask(cleaned)
    cleaned["_known_artifact"] = artifact_mask.astype(int)
    if drop_known_artifacts:
        cleaned = cleaned.loc[~artifact_mask].copy()
    numeric_columns = [c for c in numeric_columns if c in cleaned.columns]
    if numeric_columns:
        cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(method="linear", limit_direction="both")
    return cleaned.reset_index(drop=True)


def attach_metadata(experiment: pd.DataFrame, metadata: pd.DataFrame, experiment_id: int) -> pd.DataFrame:
    row = metadata.loc[metadata["No"] == experiment_id]
    if row.empty:
        raise ValueError(f"No metadata row for experiment {experiment_id}")
    if len(row) != 1:
        raise ValueError(f"Multiple metadata rows for experiment {experiment_id}")
    result = experiment.copy()
    result["experiment_id"] = experiment_id
    for column in row.columns:
        if column != "No":
            result[column] = row.iloc[0][column]
    return result


def load_all_experiments(raw_dir: str | Path, metadata_file: str = "train.csv", pattern: str = "experiment_*.csv", *, drop_known_artifacts: bool = True) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    raw_dir = Path(raw_dir)
    metadata = load_metadata(raw_dir / metadata_file)
    experiments: list[pd.DataFrame] = []
    for path in discover_experiment_files(raw_dir, pattern):
        experiment_id = experiment_id_from_path(path)
        frame = clean_experiment(load_experiment(path), drop_known_artifacts=drop_known_artifacts)
        experiments.append(attach_metadata(frame, metadata, experiment_id))
    return metadata, experiments


def validate_dataset(raw_dir: str | Path, *, metadata_file: str = "train.csv", pattern: str = "experiment_*.csv", expected_experiments: int = 18) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    metadata = load_metadata(raw_dir / metadata_file)
    files = discover_experiment_files(raw_dir, pattern)
    ids = [experiment_id_from_path(path) for path in files]
    expected_ids = list(range(1, expected_experiments + 1))
    if ids != expected_ids:
        raise ValueError(f"Expected experiment ids {expected_ids}; found {ids}")
    if len(metadata) != expected_experiments:
        raise ValueError(f"Expected {expected_experiments} metadata rows; found {len(metadata)}")

    row_counts: dict[int, int] = {}
    column_counts: dict[int, int] = {}
    schema: set[str] | None = None
    for path in files:
        experiment_id = experiment_id_from_path(path)
        frame = load_experiment(path)
        row_counts[experiment_id] = len(frame)
        column_counts[experiment_id] = len(frame.columns)
        current_schema = set(frame.columns)
        if schema is None:
            schema = current_schema
        elif current_schema != schema:
            raise ValueError(f"Schema mismatch detected in experiment {experiment_id}")

    if schema is None or len(schema) < 40:
        raise ValueError("Expected a high-dimensional CNC telemetry schema (>=40 columns).")
    required_targets = {"tool_condition", "machining_finalized", "passed_visual_inspection"}
    missing_targets = required_targets - set(metadata.columns)
    if missing_targets:
        raise ValueError(f"Missing metadata targets: {sorted(missing_targets)}")
    return {
        "experiments": len(files),
        "metadata_rows": len(metadata),
        "sensor_columns": len(schema),
        "total_time_series_rows": int(sum(row_counts.values())),
        "rows_per_experiment": row_counts,
        "columns_per_experiment": column_counts,
        "tool_condition_counts": metadata["tool_condition"].value_counts(dropna=False).to_dict(),
    }
