from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _canon(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


ALIASES: dict[str, tuple[str, ...]] = {
    "Run": ("Run", "run id", "run number"),
    "Exp": ("Exp", "experiment"),
    "Tool": ("Tool", "tool id"),
    "Block": ("Block",),
    "SBlock": ("SBlock", "sub block", "subblock"),
    "Position": ("Position", "roughness position", "measurement position"),
    "Condition": ("Condition",),
    "TCond": ("TCond", "tool condition", "VBB", "flank wear", "flank wear width"),
    "ap": ("ap", "depth of cut"),
    "vc": ("vc", "cutting speed"),
    "f": ("f", "feed", "feed rate"),
    "Ra": ("Ra",),
    "Rsk": ("Rsk",),
    "Rku": ("Rku",),
    "RSm": ("RSm",),
    "Rt": ("Rt",),
    "Fx": ("Fx", "Fc", "cutting force"),
    "Fy": ("Fy", "passive force"),
    "Fz": ("Fz", "feed force"),
    "F": ("F", "resultant force", "force resultant"),
}


def resolve_column(columns: Iterable[str], key: str, *, required: bool = True) -> str | None:
    columns = [str(c) for c in columns]
    exact = {str(c).strip(): str(c) for c in columns}
    aliases = ALIASES.get(key, (key,))
    # Preserve the source dataset's meaningful case distinction between feed `f`
    # and resultant force `F` before falling back to case-insensitive matching.
    for alias in aliases:
        hit = exact.get(str(alias).strip())
        if hit is not None:
            return hit
    lookup = {_canon(c): str(c) for c in columns}
    for alias in aliases:
        hit = lookup.get(_canon(alias))
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"Could not resolve '{key}' from columns: {list(columns)}")
    return None


def normalize_experiment(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]

    required = ["Run", "ap", "vc", "f", "Ra"]
    optional = [
        "Exp", "Tool", "Block", "SBlock", "Position", "Condition", "TCond",
        "Rsk", "Rku", "RSm", "Rt", "Fx", "Fy", "Fz", "F",
    ]
    rename: dict[str, str] = {}
    for key in required:
        source = resolve_column(out.columns, key, required=True)
        if source != key:
            rename[source] = key
    for key in optional:
        source = resolve_column(out.columns, key, required=False)
        if source is not None and source != key:
            rename[source] = key
    out = out.rename(columns=rename)

    out["experiment"] = experiment
    if "TCond" not in out.columns:
        out["TCond"] = 0.0
    if "Position" not in out.columns:
        out["Position"] = np.arange(len(out), dtype=int)

    numeric = [
        "Run", "ap", "vc", "f", "TCond", "Ra", "Rsk", "Rku", "RSm", "Rt",
        "Fx", "Fy", "Fz", "F", "Position",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "F" not in out.columns or out["F"].isna().all():
        components = [c for c in ["Fx", "Fy", "Fz"] if c in out.columns]
        if len(components) == 3:
            out["F"] = np.sqrt((out[components].astype(float) ** 2).sum(axis=1))

    if out["Run"].isna().any():
        raise ValueError(f"{experiment}: Run contains missing/non-numeric values after parsing")
    return out


def load_experiment(path: str | Path, experiment: str) -> pd.DataFrame:
    return normalize_experiment(pd.read_csv(path), experiment)


def _first_nonmissing(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[0] if len(valid) else np.nan


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment", "Run", "ap", "vc", "f", "TCond", "Ra"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns for run aggregation: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (experiment, run), group in frame.groupby(["experiment", "Run"], sort=True, dropna=False):
        record: dict[str, object] = {
            "experiment": str(experiment),
            "Run": int(run),
            "run_key": f"{experiment}:{int(run)}",
            "n_positions": int(len(group)),
            "n_unique_positions": int(group["Position"].nunique(dropna=True)) if "Position" in group else int(len(group)),
        }

        for col in ["ap", "vc", "f", "TCond", "Fx", "Fy", "Fz", "F"]:
            if col in group.columns:
                record[col] = float(group[col].median(skipna=True)) if group[col].notna().any() else np.nan

        for col in ["Ra", "Rsk", "Rku", "RSm", "Rt"]:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce")
                record[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                record[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0

        for col in ["Tool", "Block", "SBlock", "Condition"]:
            if col in group.columns:
                record[col] = _first_nonmissing(group[col])

        rows.append(record)

    result = pd.DataFrame(rows)
    result["TCond"] = pd.to_numeric(result.get("TCond", 0.0), errors="coerce").fillna(0.0)
    result["mrr_proxy"] = result["ap"] * result["f"] * result["vc"]
    result["condition_key"] = result.apply(
        lambda r: "|".join(
            f"{float(r[c]):.8g}" if pd.notna(r[c]) else "nan"
            for c in ["ap", "vc", "f", "TCond"]
        ),
        axis=1,
    )
    return result.sort_values(["experiment", "Run"]).reset_index(drop=True)


def load_run_level(raw_dir: str | Path, exp1_file: str = "Exp1.csv", exp2_file: str = "Exp2.csv") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    exp1 = load_experiment(raw_dir / exp1_file, "Exp1")
    exp2 = load_experiment(raw_dir / exp2_file, "Exp2")
    return aggregate_runs(pd.concat([exp1, exp2], ignore_index=True))


def available_force_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["Fx", "Fy", "Fz", "F"] if c in frame.columns and frame[c].notna().any()]
