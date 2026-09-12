from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CSV = RAW / "extrusion.csv"
LEGEND = RAW / "legend.xlsx"

TERMS = [
    "riss", "abriss", "film", "folie", "folien", "break", "rupt", "fault",
    "fehler", "stoer", "stör", "alarm", "warn", "defect", "abzug", "haul",
    "dicke", "thick", "bahn", "web", "bubble", "blasen", "stop", "stopp",
]


def match_text(value: object) -> bool:
    text = str(value).lower()
    return any(term in text for term in TERMS)


def main() -> None:
    if not CSV.exists() or not LEGEND.exists():
        raise FileNotFoundError("Run scripts/download_data.py first")

    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    matching_columns = [c for c in header if match_text(c)]

    book = pd.read_excel(LEGEND, sheet_name=None, header=None)
    legend_hits: list[dict[str, object]] = []
    for sheet, df in book.items():
        for ridx, row in df.iterrows():
            values = [None if pd.isna(v) else v for v in row.tolist()]
            if any(match_text(v) for v in values if v is not None):
                legend_hits.append({"sheet": str(sheet), "row": int(ridx), "values": values})

    thick = [c for c in header if re.search(r"ST110_VAREx_[0-3]_SDickeIst$", c)]
    output = [c for c in header if re.search(r"ST110_VAREx_[0-3]_GesamtDS$", c)]
    controls = [
        c for c in header
        if c in {
            "ST0_VARActAuftrag",
            "ST110_VARAbzug_IstSpeed",
            "ST110_VARAbzugIstSpeed",
            "ST113_VARLmpRun",
            "ST114_VARLmpRun",
        }
        or re.search(r"ST110_VAREx_[0-3]_RegelungEin$", c)
    ]
    # Include every haul-off / line-speed candidate for inspection.
    controls += [c for c in header if ("Abzug" in c or "IstSpeed" in c) and c not in controls]
    controls = controls[:40]

    usecols = ["Datum", *thick, *output, *controls]
    usecols = list(dict.fromkeys(c for c in usecols if c in header))
    frame = pd.read_csv(CSV, usecols=usecols, low_memory=False)
    ts = pd.to_datetime(frame["Datum"], format="%d.%m.%Y %H:%M", errors="coerce")
    gap = ts.diff().dt.total_seconds()

    numeric_cols = [c for c in usecols if c != "Datum"]
    for c in numeric_cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    thickness = frame[thick] if thick else pd.DataFrame(index=frame.index)
    all_zero = (thickness.fillna(np.nan) == 0).all(axis=1) if thick else pd.Series(False, index=frame.index)
    all_positive = (thickness > 0).all(axis=1) if thick else pd.Series(False, index=frame.index)
    any_zero = (thickness == 0).any(axis=1) if thick else pd.Series(False, index=frame.index)

    output_sum = frame[output].sum(axis=1, min_count=1) if output else pd.Series(np.nan, index=frame.index)
    output_positive = output_sum > 0

    onset = all_zero & ~all_zero.shift(fill_value=False)
    strict_onset = onset & all_positive.shift(fill_value=False)
    output_onset = onset & output_positive.shift(fill_value=False)

    # Characterize all-zero episode length in rows and wall-clock time.
    episode_id = onset.cumsum().where(all_zero)
    episodes = []
    for eid, idx in episode_id.dropna().groupby(episode_id.dropna()).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        start = int(pos.min())
        end = int(pos.max())
        episodes.append({
            "episode": int(eid),
            "start_row": start,
            "end_row": end,
            "rows": int(end - start + 1),
            "start": ts.iloc[start].isoformat() if pd.notna(ts.iloc[start]) else None,
            "end": ts.iloc[end].isoformat() if pd.notna(ts.iloc[end]) else None,
            "wall_minutes": float((ts.iloc[end] - ts.iloc[start]).total_seconds() / 60.0) if pd.notna(ts.iloc[start]) and pd.notna(ts.iloc[end]) else None,
            "prev_all_positive": bool(all_positive.iloc[start - 1]) if start > 0 else False,
            "prev_output_positive": bool(output_positive.iloc[start - 1]) if start > 0 else False,
            "gap_from_prev_seconds": float(gap.iloc[start]) if start > 0 and pd.notna(gap.iloc[start]) else None,
        })

    # Snapshot selected control values immediately before strict onsets.
    onset_rows = np.flatnonzero(strict_onset.to_numpy())[:100]
    onset_preview = []
    for row in onset_rows:
        prev = row - 1
        rec = {
            "row": int(row),
            "timestamp": ts.iloc[row].isoformat() if pd.notna(ts.iloc[row]) else None,
            "gap_seconds": float(gap.iloc[row]) if pd.notna(gap.iloc[row]) else None,
        }
        for c in [*thick, *output, *controls[:15]]:
            if c in frame.columns:
                v = frame[c].iloc[prev]
                rec[f"prev::{c}"] = None if pd.isna(v) else float(v)
        onset_preview.append(rec)

    gap_valid = gap.dropna()
    result = {
        "matching_columns": matching_columns,
        "legend_hits": legend_hits,
        "thickness_columns": thick,
        "output_columns": output,
        "control_candidates": controls,
        "timestamp": {
            "rows": int(len(frame)),
            "duplicates": int(ts.duplicated().sum()),
            "monotonic": bool(ts.is_monotonic_increasing),
            "gap_seconds_quantiles": {
                str(q): float(gap_valid.quantile(q)) for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            },
            "gaps_gt_5min": int((gap_valid > 300).sum()),
            "gaps_gt_60min": int((gap_valid > 3600).sum()),
        },
        "thickness_zero_patterns": {
            "all_zero_rows": int(all_zero.sum()),
            "any_zero_rows": int(any_zero.sum()),
            "all_positive_rows": int(all_positive.sum()),
            "all_zero_onsets": int(onset.sum()),
            "strict_positive_to_all_zero_onsets": int(strict_onset.sum()),
            "positive_output_to_all_zero_onsets": int(output_onset.sum()),
        },
        "episodes_summary": {
            "episodes": int(len(episodes)),
            "strict_prev_positive": int(sum(int(e["prev_all_positive"]) for e in episodes)),
            "prev_output_positive": int(sum(int(e["prev_output_positive"]) for e in episodes)),
            "first_80": episodes[:80],
        },
        "strict_onset_preview": onset_preview,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
