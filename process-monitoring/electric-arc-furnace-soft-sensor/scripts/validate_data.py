from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_snapshot_dataset, load_tables


def main() -> None:
    raw = ROOT / "data" / "raw"
    tables = load_tables(raw)
    snapshots = build_snapshot_dataset(tables)
    if snapshots.empty:
        raise ValueError("No leakage-safe temperature snapshots could be constructed")
    if len(snapshots) < 40:
        raise ValueError(f"Too few snapshots for chronological evaluation: {len(snapshots)}")
    if not (snapshots["target_time"] > snapshots["snapshot_time"]).all():
        raise ValueError("Found non-future target timestamp")

    summary = {
        "table_rows": {name: int(len(frame)) for name, frame in tables.items()},
        "unique_heats_by_table": {name: int(frame["HEATID"].nunique()) for name, frame in tables.items()},
        "snapshot_rows": int(len(snapshots)),
        "snapshot_heats": int(snapshots["HEATID"].nunique()),
        "target_time_min": snapshots["target_time"].min().isoformat(),
        "target_time_max": snapshots["target_time"].max().isoformat(),
        "median_forecast_horizon_min": float(snapshots["forecast_horizon_min"].median()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
