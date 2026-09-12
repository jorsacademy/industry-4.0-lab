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

STRONG_TERMS = ["riss", "abriss", "break", "rupt", "fault", "fehler", "stoer", "stör", "alarm", "defect"]
PROCESS_TERMS = ["film", "folie", "folien", "bahn", "web", "bubble", "blasen", "dicke", "thick", "abzug", "haul"]


def hits(value: object, terms: list[str]) -> bool:
    text = str(value).lower()
    return any(term in text for term in terms)


def main() -> None:
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    strong_columns = [c for c in header if hits(c, STRONG_TERMS)]
    process_columns = [c for c in header if hits(c, PROCESS_TERMS)]

    book = pd.read_excel(LEGEND, sheet_name=None, header=None)
    strong_legend: list[dict[str, object]] = []
    film_legend: list[dict[str, object]] = []
    for sheet, df in book.items():
        for ridx, row in df.iterrows():
            values = [None if pd.isna(v) else v for v in row.tolist()]
            joined = " | ".join(str(v) for v in values if v is not None)
            if hits(joined, STRONG_TERMS):
                strong_legend.append({"sheet": str(sheet), "row": int(ridx), "values": values})
            if hits(joined, ["film", "folie", "folien", "bahn", "web", "bubble", "blasen"]):
                film_legend.append({"sheet": str(sheet), "row": int(ridx), "values": values})

    thick = [c for c in header if re.search(r"ST110_VAREx_[0-3]_SDickeIst$", c)]
    output = [c for c in header if re.search(r"ST110_VAREx_[0-3]_GesamtDS$", c)]
    controls = [c for c in header if c in {"ST0_VARActAuftrag", "ST113_VARLmpRun", "ST114_VARLmpRun"}]
    controls += [c for c in header if re.search(r"ST110_VAREx_[0-3]_RegelungEin$", c)]
    controls += [c for c in header if ("Abzug" in c or "IstSpeed" in c) and c not in controls]
    controls = controls[:25]

    usecols = list(dict.fromkeys(["Datum", *thick, *output, *controls]))
    frame = pd.read_csv(CSV, usecols=usecols, low_memory=False)
    ts = pd.to_datetime(frame["Datum"], format="%d.%m.%Y %H:%M", errors="coerce")
    gap = ts.diff().dt.total_seconds()
    for c in usecols:
        if c != "Datum":
            frame[c] = pd.to_numeric(frame[c], errors="coerce")

    thickness = frame[thick]
    all_zero = (thickness == 0).all(axis=1)
    all_positive = (thickness > 0).all(axis=1)
    any_zero = (thickness == 0).any(axis=1)
    output_sum = frame[output].sum(axis=1, min_count=1) if output else pd.Series(np.nan, index=frame.index)
    output_positive = output_sum > 0
    onset = all_zero & ~all_zero.shift(fill_value=False)
    strict_onset = onset & all_positive.shift(fill_value=False)
    output_onset = onset & output_positive.shift(fill_value=False)

    episode_id = onset.cumsum().where(all_zero)
    episodes = []
    for eid, idx in episode_id.dropna().groupby(episode_id.dropna()).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        start, end = int(pos.min()), int(pos.max())
        episodes.append({
            "episode": int(eid), "start_row": start, "rows": int(end - start + 1),
            "start": ts.iloc[start].isoformat(), "end": ts.iloc[end].isoformat(),
            "wall_minutes": float((ts.iloc[end] - ts.iloc[start]).total_seconds() / 60),
            "prev_all_positive": bool(all_positive.iloc[start - 1]) if start else False,
            "prev_output_positive": bool(output_positive.iloc[start - 1]) if start else False,
            "gap_from_prev_seconds": float(gap.iloc[start]) if start and pd.notna(gap.iloc[start]) else None,
        })

    previews = []
    for row in np.flatnonzero(strict_onset.to_numpy())[:20]:
        prev = row - 1
        rec = {"row": int(row), "timestamp": ts.iloc[row].isoformat(), "gap_seconds": float(gap.iloc[row])}
        for c in [*thick, *output, *controls[:12]]:
            v = frame[c].iloc[prev]
            rec[c] = None if pd.isna(v) else float(v)
        previews.append(rec)

    g = gap.dropna()
    result = {
        "strong_fault_columns": strong_columns,
        "strong_fault_legend_hits": strong_legend,
        "film_web_legend_hits": film_legend[:40],
        "process_columns_matching_film_thickness_haul": process_columns,
        "thickness_columns": thick,
        "output_columns": output,
        "control_candidates": controls,
        "timestamp": {
            "duplicates": int(ts.duplicated().sum()), "monotonic": bool(ts.is_monotonic_increasing),
            "gap_quantiles_seconds": {str(q): float(g.quantile(q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]},
            "gaps_gt_5min": int((g > 300).sum()), "gaps_gt_60min": int((g > 3600).sum()),
        },
        "zero_patterns": {
            "all_zero_rows": int(all_zero.sum()), "any_zero_rows": int(any_zero.sum()),
            "all_positive_rows": int(all_positive.sum()), "all_zero_onsets": int(onset.sum()),
            "strict_positive_to_all_zero_onsets": int(strict_onset.sum()),
            "positive_output_to_all_zero_onsets": int(output_onset.sum()),
        },
        "episodes": {
            "count": len(episodes), "strict_prev_positive": sum(int(e["prev_all_positive"]) for e in episodes),
            "prev_output_positive": sum(int(e["prev_output_positive"]) for e in episodes), "first_20": episodes[:20],
        },
        "strict_onset_preview": previews,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
