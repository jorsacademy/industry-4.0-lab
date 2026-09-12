from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schema import output_columns

CSV = ROOT / "data" / "raw" / "continuous_factory_process.csv"


def main() -> None:
    if not CSV.exists():
        raise FileNotFoundError(CSV)
    frame = pd.read_csv(CSV, low_memory=False)
    stage1_actual = output_columns(frame.columns, stage=1, kind="actual")
    stage1_setpoint = output_columns(frame.columns, stage=1, kind="setpoint")
    stage2_actual = output_columns(frame.columns, stage=2, kind="actual")
    stage2_setpoint = output_columns(frame.columns, stage=2, kind="setpoint")

    if len(frame) != 14088:
        raise ValueError(f"Expected 14,088 rows, got {len(frame)}")
    if frame.shape[1] != 116:
        raise ValueError(f"Expected 116 columns, got {frame.shape[1]}")
    if "time_stamp" not in frame.columns:
        raise ValueError("Missing time_stamp")
    if any(len(cols) != 15 for cols in [stage1_actual, stage1_setpoint, stage2_actual, stage2_setpoint]):
        raise ValueError("Expected 15 actual and 15 setpoint channels at both output stages")

    ts = pd.to_datetime(frame["time_stamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Unparsable timestamps")
    if not ts.is_monotonic_increasing:
        raise ValueError("Expected chronological source rows")

    summary = {
        "rows": len(frame),
        "columns": frame.shape[1],
        "start": ts.iloc[0].isoformat(),
        "end": ts.iloc[-1].isoformat(),
        "stage1_actual_outputs": len(stage1_actual),
        "stage1_setpoints": len(stage1_setpoint),
        "stage2_actual_outputs": len(stage2_actual),
        "stage2_setpoints": len(stage2_setpoint),
        "stage2_rows_with_nonpositive_setpoint": int((frame[stage2_setpoint] <= 0).any(axis=1).sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
