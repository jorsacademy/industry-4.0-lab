from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import feature_columns, load_secom


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    df = load_secom(raw_dir / data_cfg["data_file"], raw_dir / data_cfg["labels_file"])
    checks = {
        "rows": int(len(df)), "raw_process_variables": int(len(feature_columns(df))),
        "failures": int(df["is_fail"].sum()), "passes": int((df["is_fail"] == 0).sum()),
        "parsed_timestamps": int(df["timestamp"].notna().sum()),
        "timestamp_min": df["timestamp"].min().isoformat(), "timestamp_max": df["timestamp"].max().isoformat(),
        "missing_cells": int(df[feature_columns(df)].isna().sum().sum()),
    }
    print(checks)
    assert checks["rows"] == int(data_cfg["expected_rows"])
    assert checks["raw_process_variables"] == int(data_cfg["expected_raw_features"])
    assert checks["failures"] == int(data_cfg["expected_failures"])
    assert checks["parsed_timestamps"] == checks["rows"]
    assert set(df["raw_label"].unique()) == {-1, 1}


if __name__ == "__main__":
    main()
