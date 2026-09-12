from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except (UnicodeError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not parse {path}. Tried UTF-16 and UTF-8 variants: {errors}")


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_column(columns: Iterable[str], requested: str, aliases: Iterable[str] = ()) -> str:
    mapping = {_canonical(c): str(c) for c in columns}
    for candidate in (requested, *aliases):
        hit = mapping.get(_canonical(candidate))
        if hit is not None:
            return hit
    raise KeyError(f"Required column '{requested}' not found. Available columns: {list(columns)}")


def measurement_columns(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns if re.fullmatch(r"F\d+", str(c).strip(), flags=re.I)]
    return sorted(cols, key=lambda c: int(re.search(r"\d+", c).group()))


def clean_electrical_report(
    df: pd.DataFrame,
    *,
    target_column: str = "Result",
    group_column: str = "LOT",
    time_column: str = "Time",
    pass_code: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    time_col = resolve_column(out.columns, time_column, aliases=("timestamp", "datetime", "date time"))
    target_col = resolve_column(out.columns, target_column, aliases=("status", "test result"))
    group_col = resolve_column(out.columns, group_column, aliases=("lot no", "lot number", "batch", "batch id"))

    first_col = out.columns[0]
    repeated_header = out[first_col].astype(str).str.strip().eq(first_col)
    repeated_header |= out[time_col].astype(str).str.strip().eq(time_col)
    out = out.loc[~repeated_header].copy()

    measurements = measurement_columns(out)
    if not measurements:
        raise ValueError("No F1..Fn electrical measurement columns were detected.")

    for col in [target_col, *measurements]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out[group_col] = out[group_col].astype("string").str.strip()
    out["timestamp"] = pd.to_datetime(out[time_col], errors="coerce")
    out["is_defect"] = out[target_col].ne(pass_code) & out[target_col].notna()
    out["source_row"] = out.index.astype(int)

    if group_col != "LOT":
        out["LOT"] = out[group_col]
    if target_col != "Result":
        out["Result"] = out[target_col]
    if time_col != "Time":
        out["Time"] = out[time_col]
    return out.reset_index(drop=True)


def load_electrical_report(path: str | Path, **kwargs) -> pd.DataFrame:
    return clean_electrical_report(_read_csv_with_fallback(path), **kwargs)


def feature_target_group(df: pd.DataFrame, pass_code: int = 1):
    features = measurement_columns(df)
    if len(features) < 2:
        raise ValueError("At least two electrical measurement columns are required.")
    target_col = resolve_column(df.columns, "Result", aliases=("status", "test result"))
    group_col = resolve_column(df.columns, "LOT", aliases=("lot no", "lot number", "batch", "batch id"))
    y = df[target_col].ne(pass_code).astype(int)
    groups = df[group_col].astype(str)
    return df[features].copy(), y, groups, features
