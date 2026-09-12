from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_frequency_table, validate_frequency_table


def main() -> None:
    table = load_frequency_table(ROOT / "data" / "raw")
    summary = validate_frequency_table(table)
    counts = list(summary["condition_counts"].values())
    if summary["rows"] != 2000:
        raise ValueError(f"Expected 2000 frequency windows, got {summary['rows']}")
    if summary["conditions"] != 8:
        raise ValueError(f"Expected 8 operating conditions, got {summary['conditions']}")
    if summary["acceleration_features"] != 69:
        raise ValueError(f"Expected 69 structure-borne features, got {summary['acceleration_features']}")
    if summary["audio_features"] != 100:
        raise ValueError(f"Expected 100 airborne-sound features, got {summary['audio_features']}")
    if sorted(counts) != [250] * 8:
        raise ValueError(f"Expected 250 windows per condition, got {summary['condition_counts']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
