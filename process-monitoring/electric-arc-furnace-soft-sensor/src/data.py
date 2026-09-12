from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLE_SPECS = {
    "temperature": ("eaf_temp.csv", "DATETIME"),
    "transformer": ("eaf_transformer.csv", "STARTTIME"),
    "gas": ("eaf_gaslance_mat.csv", "REVTIME"),
    "carbon": ("inj_mat.csv", "REVTIME"),
    "basket": ("basket_charged.csv", "DATETIME"),
    "added": ("eaf_added_materials.csv", "DATETIME"),
}


def normalize_heatid(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def to_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().mean() < 0.8:
        fallback = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
        out = out.fillna(fallback)
    return out


def parse_timestamp(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    first = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=False)
    if first.notna().mean() < 0.8:
        second = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=True)
        first = first.fillna(second)
    return first


def read_table(path: Path, time_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    if "HEATID" not in frame.columns:
        raise KeyError(f"{path.name}: missing HEATID")
    if time_col not in frame.columns:
        raise KeyError(f"{path.name}: missing {time_col}")
    frame["HEATID"] = normalize_heatid(frame["HEATID"])
    frame["_TS"] = parse_timestamp(frame[time_col])
    return frame


def load_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw = Path(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, (filename, time_col) in TABLE_SPECS.items():
        path = raw / filename
        if not path.exists():
            raise FileNotFoundError(path)
        tables[key] = read_table(path, time_col)
    return tables


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return to_numeric(frame[column])


def _last_valid(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) else np.nan


def _delta(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 2:
        return float(valid.iloc[-1]) if len(valid) else 0.0
    return float(valid.iloc[-1] - valid.iloc[0])


def _group_by_heat(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = frame[frame["_TS"].notna()].sort_values(["HEATID", "_TS"])
    return {str(heatid): group for heatid, group in ordered.groupby("HEATID", sort=False)}


def _before(grouped: dict[str, pd.DataFrame], heatid: str, snapshot: pd.Timestamp) -> pd.DataFrame:
    group = grouped.get(str(heatid))
    if group is None:
        return pd.DataFrame()
    return group[group["_TS"] <= snapshot]


def _event_min_time(grouped_tables: dict[str, dict[str, pd.DataFrame]], heatid: str, snapshot: pd.Timestamp):
    times: list[pd.Timestamp] = []
    for grouped in grouped_tables.values():
        subset = _before(grouped, heatid, snapshot)
        if len(subset):
            times.append(subset["_TS"].iloc[0])
    return min(times) if times else pd.NaT


def build_snapshot_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prepared = {name: frame.copy() for name, frame in tables.items()}
    temp = prepared["temperature"]
    temp["TEMP"] = to_numeric(temp["TEMP"])
    if "VALO2_PPM" in temp.columns:
        temp["VALO2_PPM"] = to_numeric(temp["VALO2_PPM"])
    prepared["temperature"] = temp
    grouped_tables = {name: _group_by_heat(frame) for name, frame in prepared.items()}

    rows: list[dict[str, object]] = []
    for heatid, group in grouped_tables["temperature"].items():
        g = group[group["TEMP"].notna()].sort_values("_TS")
        if len(g) < 2:
            continue
        target = g.iloc[-1]
        snapshot_row = g.iloc[-2]
        snapshot = snapshot_row["_TS"]
        target_time = target["_TS"]
        if pd.isna(snapshot) or pd.isna(target_time) or target_time <= snapshot:
            continue

        history_temp = g[g["_TS"] <= snapshot]
        positive_o2 = pd.Series(dtype=float)
        if "VALO2_PPM" in history_temp.columns:
            positive_o2 = history_temp.loc[history_temp["VALO2_PPM"] > 0, "VALO2_PPM"]

        record: dict[str, object] = {
            "HEATID": str(heatid),
            "snapshot_time": snapshot,
            "target_time": target_time,
            "target_temp": float(target["TEMP"]),
            "snapshot_temp": float(snapshot_row["TEMP"]),
            "snapshot_o2_ppm": _last_valid(positive_o2),
            "prior_temp_measurements": int(len(history_temp)),
            "forecast_horizon_min": float((target_time - snapshot).total_seconds() / 60.0),
        }

        earliest = _event_min_time(grouped_tables, str(heatid), snapshot)
        record["elapsed_to_snapshot_min"] = (
            float((snapshot - earliest).total_seconds() / 60.0) if pd.notna(earliest) else np.nan
        )

        transformer = _before(grouped_tables["transformer"], str(heatid), snapshot)
        mw = _num(transformer, "MW")
        duration = _num(transformer, "DURATION")
        tap = _num(transformer, "TAP")
        record.update({
            "transformer_segments": int(len(transformer)),
            "transformer_mw_sum": float(mw.sum(skipna=True)) if len(mw) else 0.0,
            "transformer_mw_mean": float(mw.mean(skipna=True)) if mw.notna().any() else np.nan,
            "transformer_duration_sum": float(duration.sum(skipna=True)) if len(duration) else 0.0,
            "transformer_tap_mean": float(tap.mean(skipna=True)) if tap.notna().any() else np.nan,
            "transformer_tap_last": _last_valid(tap),
        })

        gas = _before(grouped_tables["gas"], str(heatid), snapshot)
        o2_amount = _num(gas, "O2_AMOUNT")
        gas_amount = _num(gas, "GAS_AMOUNT")
        o2_flow = _num(gas, "O2_FLOW")
        gas_flow = _num(gas, "GAS_FLOW")
        record.update({
            "gas_events": int(len(gas)),
            "o2_amount_last": _last_valid(o2_amount),
            "o2_amount_delta": _delta(o2_amount),
            "gas_amount_last": _last_valid(gas_amount),
            "gas_amount_delta": _delta(gas_amount),
            "o2_flow_mean": float(o2_flow.mean(skipna=True)) if o2_flow.notna().any() else np.nan,
            "o2_flow_max": float(o2_flow.max(skipna=True)) if o2_flow.notna().any() else np.nan,
            "gas_flow_mean": float(gas_flow.mean(skipna=True)) if gas_flow.notna().any() else np.nan,
            "gas_flow_max": float(gas_flow.max(skipna=True)) if gas_flow.notna().any() else np.nan,
        })

        carbon = _before(grouped_tables["carbon"], str(heatid), snapshot)
        c_amount = _num(carbon, "INJ_AMOUNT_CARBON")
        c_flow = _num(carbon, "INJ_FLOW_CARBON")
        record.update({
            "carbon_events": int(len(carbon)),
            "carbon_amount_last": _last_valid(c_amount),
            "carbon_amount_delta": _delta(c_amount),
            "carbon_flow_mean": float(c_flow.mean(skipna=True)) if c_flow.notna().any() else np.nan,
            "carbon_flow_max": float(c_flow.max(skipna=True)) if c_flow.notna().any() else np.nan,
        })

        for source, prefix in [("basket", "basket"), ("added", "added")]:
            material = _before(grouped_tables[source], str(heatid), snapshot)
            amount = _num(material, "CHARGE_AMOUNT")
            record[f"{prefix}_events"] = int(len(material))
            record[f"{prefix}_charge_total"] = float(amount.sum(skipna=True)) if len(amount) else 0.0
            record[f"{prefix}_unique_materials"] = (
                int(material["MAT_CODE"].nunique(dropna=True)) if "MAT_CODE" in material.columns else 0
            )

        rows.append(record)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["target_time", "HEATID"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"HEATID", "snapshot_time", "target_time", "target_temp"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
