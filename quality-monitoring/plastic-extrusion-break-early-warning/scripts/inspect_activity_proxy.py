from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "extrusion.csv"

THICK = [f"ST110_VAREx_{i}_SDickeIst" for i in range(4)]
OUTPUT = [f"ST110_VAREx_{i}_GesamtDS" for i in range(4)]
WINDERS = ["ST113_VARLmpRun", "ST114_VARLmpRun"]
HAUL_ON = "ST112_VARAbzug_1_IstEin"
HAUL_CLOSED = "ST112_VARAbzug_1_IstZu"
HAUL_SPEED = "ST112_VARAbzug_1_SollSpeed"


def summarize(mask: pd.Series, onset: pd.Series, ts: pd.Series, gap: pd.Series) -> dict[str, object]:
    event = onset & mask.shift(1, fill_value=False) & gap.le(180)
    idx = np.flatnonzero(event.to_numpy())
    n = len(ts)
    blocks = [
        ("fit55", 0.00, 0.55),
        ("selection15", 0.55, 0.70),
        ("calibration15", 0.70, 0.85),
        ("test15", 0.85, 1.00),
    ]
    by_block = {}
    for name, lo, hi in blocks:
        a, b = int(np.floor(lo * n)), int(np.floor(hi * n)) if hi < 1 else n
        block_mask = mask.iloc[a:b]
        block_event = event.iloc[a:b]
        by_block[name] = {
            "rows": int(b - a),
            "active_rows": int(block_mask.sum()),
            "events": int(block_event.sum()),
            "start": ts.iloc[a].isoformat() if a < n else None,
            "end": ts.iloc[b - 1].isoformat() if b > a else None,
        }
    return {
        "active_rows": int(mask.sum()),
        "event_onsets": int(event.sum()),
        "event_rows_first20": idx[:20].tolist(),
        "event_times_first20": [ts.iloc[i].isoformat() for i in idx[:20]],
        "by_block": by_block,
    }


def main() -> None:
    usecols = ["Datum", *THICK, *OUTPUT, *WINDERS, HAUL_ON, HAUL_CLOSED, HAUL_SPEED]
    df = pd.read_csv(CSV, usecols=usecols, low_memory=False)
    ts = pd.to_datetime(df["Datum"], format="%d.%m.%Y %H:%M", errors="raise")
    gap = ts.diff().dt.total_seconds()
    for c in usecols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    all_pos_thick = (df[THICK] > 0).all(axis=1)
    all_zero_thick = (df[THICK] == 0).all(axis=1)
    onset = all_zero_thick & ~all_zero_thick.shift(fill_value=False)
    output_sum = df[OUTPUT].sum(axis=1, min_count=4)
    output_all_pos = (df[OUTPUT] > 0).all(axis=1)
    one_winder = df[WINDERS].eq(1).any(axis=1)
    both_winders = df[WINDERS].eq(1).all(axis=1)
    haul = df[HAUL_ON].eq(1) & df[HAUL_CLOSED].eq(1)

    candidates = {
        "A_thickness_output": all_pos_thick & output_all_pos,
        "B_plus_haul_speed10": all_pos_thick & output_all_pos & haul & df[HAUL_SPEED].ge(10),
        "C_plus_any_winder": all_pos_thick & output_all_pos & haul & df[HAUL_SPEED].ge(10) & one_winder,
        "D_plus_both_winders": all_pos_thick & output_all_pos & haul & df[HAUL_SPEED].ge(10) & both_winders,
        "E_any_winder_speed20": all_pos_thick & output_all_pos & haul & df[HAUL_SPEED].ge(20) & one_winder,
        "F_any_winder_speed30": all_pos_thick & output_all_pos & haul & df[HAUL_SPEED].ge(30) & one_winder,
        "G_any_winder_speed10_output100": all_pos_thick & output_all_pos & haul & df[HAUL_SPEED].ge(10) & one_winder & output_sum.ge(100),
    }

    result = {name: summarize(mask, onset, ts, gap) for name, mask in candidates.items()}
    result["source"] = {
        "rows": int(len(df)),
        "zero_onsets_total": int(onset.sum()),
        "normal_gap_event_definition_seconds": 180,
    }
    text = json.dumps(result, indent=2)
    print(text)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "activity_proxy_summary.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
