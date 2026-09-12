from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "config.yaml"


def locate_csv() -> Path:
    files = [p for p in RAW.rglob("*.csv") if p.is_file()]
    if not files:
        raise FileNotFoundError("No CSV source file found under data/raw")
    return max(files, key=lambda p: p.stat().st_size)


def safe_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    path = locate_csv()
    frame = pd.read_csv(path, low_memory=False)
    expected_rows = int(cfg["source"]["expected_rows"])
    min_cols = int(cfg["source"]["min_columns"])
    max_cols = int(cfg["source"]["max_columns"])
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(frame)}")
    if not (min_cols <= frame.shape[1] <= max_cols):
        raise ValueError(f"Unexpected column count: {frame.shape[1]}")

    numeric = safe_numeric(frame)
    numeric_fraction = numeric.notna().mean()
    mostly_numeric = numeric_fraction[numeric_fraction >= 0.95].index.tolist()
    name_candidates = [
        c for c in frame.columns
        if any(k in str(c).lower() for k in ("thick", "tolsh", "defect", "break", "rupt", "film", "width"))
    ]
    zero_stats = []
    for c in mostly_numeric:
        s = numeric[c]
        valid = s.dropna()
        if valid.empty:
            continue
        zero_fraction = float((valid == 0).mean())
        if 0 < zero_fraction < 0.25:
            zero_stats.append({
                "column": str(c),
                "zero_fraction": zero_fraction,
                "non_null_fraction": float(s.notna().mean()),
                "min": float(valid.min()),
                "median": float(valid.median()),
                "max": float(valid.max()),
                "nunique": int(valid.nunique()),
            })
    zero_stats.sort(key=lambda d: d["zero_fraction"], reverse=True)

    summary = {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "mostly_numeric_columns": int(len(mostly_numeric)),
        "name_candidates": name_candidates,
        "first_80_columns": [str(c) for c in frame.columns[:80]],
        "last_40_columns": [str(c) for c in frame.columns[-40:]],
        "zero_fraction_candidates_top40": zero_stats[:40],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "source_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
