from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import EXPECTED_RECORDINGS, REQUIRED_COLUMNS, find_member, inspect_archive


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    archive = ROOT / "data" / "raw" / cfg["source"]["archive_name"]
    summary = inspect_archive(archive)
    sample_summary: dict[str, dict[str, float | int]] = {}
    with zipfile.ZipFile(archive) as zf:
        for basename in EXPECTED_RECORDINGS:
            member = find_member(zf, basename)
            with zf.open(member) as handle:
                sample = pd.read_csv(handle, usecols=REQUIRED_COLUMNS, nrows=16384)
            numeric = sample.apply(pd.to_numeric, errors="coerce")
            if numeric[REQUIRED_COLUMNS].notna().mean().min() < 0.99:
                raise ValueError(f"{basename}: source columns are not reliably numeric")
            sample_summary[basename] = {
                "sample_rows": int(len(sample)),
                "rpm_min": float(numeric["Measured_RPM"].min()),
                "rpm_max": float(numeric["Measured_RPM"].max()),
            }
    summary["samples"] = sample_summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
