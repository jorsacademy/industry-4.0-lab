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
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


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


def _ensure_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
        else:
            out[col] = to_numeric(out[col])
    return out


def _observable(frame: pd.DataFrame, snapshot_lookup: pd.Series) -> pd.DataFrame:
    subset = frame[frame["HEATID"].isin(snapshot_lookup.index) & frame["_TS"].notna()].copy()
    subset["_SNAPSHOT"] = subset["HEATID"].map(snapshot_lookup)
    return subset[subset["_TS"] <= subset["_SNAPSHOT"]].sort_values(["HEATID", "_TS"])


def _merge_features(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return base
    return base.merge(features.reset_index(), on="HEATID", how="left")


def build_snapshot_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    temp = _ensure_numeric(tables["temperature"], ["TEMP", "VALO2_PPM"])
    temp = temp[temp["_TS"].notna() & temp["TEMP"].notna()].sort_values(["HEATID", "_TS"]).copy()
    if temp.empty:
        return pd.DataFrame()

    temp["_FROM_END"] = temp.groupby("HEATID").cumcount(ascending=False)
    target = temp[temp["_FROM_END"] == 0][["HEATID", "_TS", "TEMP"]].rename(
        columns={"_TS": "target_time", "TEMP": "target_temp"}
    )
    snap = temp[temp["_FROM_END"] == 1][["HEATID", "_TS", "TEMP"]].rename(
        columns={"_TS": "snapshot_time", "TEMP": "snapshot_temp"}
    )
    snapshots = snap.merge(target, on="HEATID", how="inner")
    snapshots = snapshots[snapshots["target_time"] > snapshots["snapshot_time"]].copy()
    if snapshots.empty:
        return snapshots

    counts = temp.groupby("HEATID").size().rename("temperature_measurements")
    snapshots["prior_temp_measurements"] = snapshots["HEATID"].map(counts).astype(int) - 1
    snapshots["forecast_horizon_min"] = (
        snapshots["target_time"] - snapshots["snapshot_time"]
    ).dt.total_seconds() / 60.0
    snapshot_lookup = snapshots.set_index("HEATID")["snapshot_time"]

    temp_history = _observable(temp.drop(columns=["_FROM_END"]), snapshot_lookup)
    positive_o2 = temp_history[temp_history["VALO2_PPM"] > 0]
    if len(positive_o2):
        latest_o2 = positive_o2.groupby("HEATID")["VALO2_PPM"].last().rename("snapshot_o2_ppm")
        snapshots = _merge_features(snapshots, latest_o2.to_frame())
    else:
        snapshots["snapshot_o2_ppm"] = np.nan

    event_min_parts = [temp_history.groupby("HEATID")["_TS"].min()]

    transformer = _observable(tables["transformer"], snapshot_lookup)
    transformer = _ensure_numeric(transformer, ["MW", "DURATION", "TAP"])
    if len(transformer):
        tr = transformer.groupby("HEATID").agg(
            transformer_segments=("_TS", "size"),
            transformer_mw_sum=("MW", "sum"),
            transformer_mw_mean=("MW", "mean"),
            transformer_duration_sum=("DURATION", "sum"),
            transformer_tap_mean=("TAP", "mean"),
            transformer_tap_last=("TAP", "last"),
        )
        snapshots = _merge_features(snapshots, tr)
        event_min_parts.append(transformer.groupby("HEATID")["_TS"].min())

    gas = _observable(tables["gas"], snapshot_lookup)
    gas = _ensure_numeric(gas, ["O2_AMOUNT", "GAS_AMOUNT", "O2_FLOW", "GAS_FLOW"])
    if len(gas):
        gg = gas.groupby("HEATID")
        gas_agg = gg.agg(
            gas_events=("_TS", "size"),
            o2_amount_first=("O2_AMOUNT", "first"),
            o2_amount_last=("O2_AMOUNT", "last"),
            gas_amount_first=("GAS_AMOUNT", "first"),
            gas_amount_last=("GAS_AMOUNT", "last"),
            o2_flow_mean=("O2_FLOW", "mean"),
            o2_flow_max=("O2_FLOW", "max"),
            gas_flow_mean=("GAS_FLOW", "mean"),
            gas_flow_max=("GAS_FLOW", "max"),
        )
        gas_agg["o2_amount_delta"] = gas_agg["o2_amount_last"] - gas_agg["o2_amount_first"]
        gas_agg["gas_amount_delta"] = gas_agg["gas_amount_last"] - gas_agg["gas_amount_first"]
        gas_agg = gas_agg.drop(columns=["o2_amount_first", "gas_amount_first"])
        snapshots = _merge_features(snapshots, gas_agg)
        event_min_parts.append(gas.groupby("HEATID")["_TS"].min())

    carbon = _observable(tables["carbon"], snapshot_lookup)
    carbon = _ensure_numeric(carbon, ["INJ_AMOUNT_CARBON", "INJ_FLOW_CARBON"])
    if len(carbon):
        cg = carbon.groupby("HEATID")
        carbon_agg = cg.agg(
            carbon_events=("_TS", "size"),
            carbon_amount_first=("INJ_AMOUNT_CARBON", "first"),
            carbon_amount_last=("INJ_AMOUNT_CARBON", "last"),
            carbon_flow_mean=("INJ_FLOW_CARBON", "mean"),
            carbon_flow_max=("INJ_FLOW_CARBON", "max"),
        )
        carbon_agg["carbon_amount_delta"] = carbon_agg["carbon_amount_last"] - carbon_agg["carbon_amount_first"]
        carbon_agg = carbon_agg.drop(columns=["carbon_amount_first"])
        snapshots = _merge_features(snapshots, carbon_agg)
        event_min_parts.append(carbon.groupby("HEATID")["_TS"].min())

    for source, prefix in [("basket", "basket"), ("added", "added")]:
        material = _observable(tables[source], snapshot_lookup)
        material = _ensure_numeric(material, ["CHARGE_AMOUNT"])
        if len(material):
            mg = material.groupby("HEATID")
            material_agg = mg.agg(
                **{
                    f"{prefix}_events": ("_TS", "size"),
                    f"{prefix}_charge_total": ("CHARGE_AMOUNT", "sum"),
                }
            )
            if "MAT_CODE" in material.columns:
                material_agg[f"{prefix}_unique_materials"] = mg["MAT_CODE"].nunique(dropna=True)
            else:
                material_agg[f"{prefix}_unique_materials"] = 0
            snapshots = _merge_features(snapshots, material_agg)
            event_min_parts.append(material.groupby("HEATID")["_TS"].min())

    earliest = pd.concat(event_min_parts, axis=1).min(axis=1).rename("_EARLIEST")
    snapshots = _merge_features(snapshots, earliest.to_frame())
    snapshots["elapsed_to_snapshot_min"] = (
        snapshots["snapshot_time"] - snapshots["_EARLIEST"]
    ).dt.total_seconds() / 60.0
    snapshots = snapshots.drop(columns=["_EARLIEST"])

    zero_fill = [
        "transformer_segments", "transformer_mw_sum", "transformer_duration_sum",
        "gas_events", "carbon_events", "basket_events", "basket_charge_total",
        "basket_unique_materials", "added_events", "added_charge_total", "added_unique_materials",
    ]
    for col in zero_fill:
        if col not in snapshots.columns:
            snapshots[col] = 0.0
        else:
            snapshots[col] = snapshots[col].fillna(0.0)

    return snapshots.sort_values(["target_time", "HEATID"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"HEATID", "snapshot_time", "target_time", "target_temp"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
