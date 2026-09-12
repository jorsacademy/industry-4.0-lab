from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import locate_source_csv, prepare_data  # noqa: E402

EXPECTED_ROWS = 226536
EXPECTED_COLUMNS = 470
EXPECTED_ACTIVE_ROWS = 193941
EXPECTED_PROXY_ONSETS = 640


def main() -> None:
    path = locate_source_csv(ROOT / "data" / "raw")
    prepared = prepare_data(path)
    frame = prepared.frame
    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected source shape: {frame.shape}")
    if prepared.timestamps.duplicated().any():
        raise ValueError("Duplicate source timestamps found")
    if int(prepared.active.sum()) != EXPECTED_ACTIVE_ROWS:
        raise ValueError(
            f"Active-state semantics changed: {int(prepared.active.sum())} != {EXPECTED_ACTIVE_ROWS}"
        )
    if int(prepared.event_onset.sum()) != EXPECTED_PROXY_ONSETS:
        raise ValueError(
            f"Interruption-proxy semantics changed: {int(prepared.event_onset.sum())} != {EXPECTED_PROXY_ONSETS}"
        )
    gap = prepared.timestamps.diff().dt.total_seconds().dropna()
    summary = {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start": prepared.timestamps.iloc[0].isoformat(),
        "end": prepared.timestamps.iloc[-1].isoformat(),
        "median_sample_gap_seconds": float(gap.median()),
        "gaps_over_5_minutes": int((gap > 300).sum()),
        "active_rows": int(prepared.active.sum()),
        "interruption_proxy_onsets": int(prepared.event_onset.sum()),
        "positive_20min_warning_rows": int(prepared.target.sum()),
        "predictor_columns_after_direct_state_exclusion": int(len(prepared.predictor_columns)),
        "explicit_fault_label_present": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
