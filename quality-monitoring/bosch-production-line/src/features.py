from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import StationKey, group_columns_by_station


@dataclass(frozen=True)
class FeatureOptions:
    station_presence: bool = True
    station_time: bool = True
    numeric_mean: bool = True
    numeric_std: bool = True
    numeric_count: bool = True
    line_station_counts: bool = True


def _row_nanmin(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, np.inf)
    out = safe.min(axis=1)
    out[~finite.any(axis=1)] = np.nan
    return out


def _row_nanmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, -np.inf)
    out = safe.max(axis=1)
    out[~finite.any(axis=1)] = np.nan
    return out


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.where(finite, values, 0.0).sum(axis=1)
    out = np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
    return out


def _row_nanstd(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    mean = _row_nanmean(values)
    centered = np.where(finite, values - mean[:, None], 0.0)
    var = np.divide((centered**2).sum(axis=1), count, out=np.full(len(values), np.nan), where=count > 0)
    return np.sqrt(var)


def _station_times(date_df: pd.DataFrame, groups: dict[StationKey, list[str]]) -> dict[StationKey, np.ndarray]:
    result: dict[StationKey, np.ndarray] = {}
    for station, columns in groups.items():
        values = date_df[columns].to_numpy(dtype=float, copy=False)
        result[station] = _row_nanmin(values)
    return result


def _global_start_end(station_times: dict[StationKey, np.ndarray], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if not station_times:
        return np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    matrix = np.column_stack(list(station_times.values()))
    return _row_nanmin(matrix), _row_nanmax(matrix)


def extract_station_features(
    numeric_df: pd.DataFrame,
    date_df: pd.DataFrame,
    *,
    prefix_fraction: float = 1.0,
    options: FeatureOptions | None = None,
) -> pd.DataFrame:
    if not (0 < prefix_fraction <= 1.0):
        raise ValueError("prefix_fraction must be in (0, 1]")
    options = options or FeatureOptions()

    if "Id" not in numeric_df or "Id" not in date_df:
        raise ValueError("Both numeric and date tables must contain Id")
    if len(numeric_df) != len(date_df):
        raise ValueError("Numeric/date chunks have different row counts")
    if not np.array_equal(numeric_df["Id"].to_numpy(), date_df["Id"].to_numpy()):
        raise ValueError("Numeric/date chunks are not aligned by Id")

    n_rows = len(numeric_df)
    numeric_groups = group_columns_by_station(numeric_df.columns, kind="F")
    date_groups = group_columns_by_station(date_df.columns, kind="D")
    station_times = _station_times(date_df, date_groups)
    start_time, final_end_time = _global_start_end(station_times, n_rows)
    final_duration = final_end_time - start_time
    cutoff = start_time + prefix_fraction * np.where(np.isfinite(final_duration), final_duration, 0.0)
    # Prefix feature tables represent a snapshot at the observation cutoff. The
    # final route end/duration are never exposed to an early-warning model.
    if prefix_fraction < 1.0:
        observed_end_time = np.where(np.isfinite(start_time), cutoff, np.nan)
        observed_duration = observed_end_time - start_time
    else:
        observed_end_time = final_end_time
        observed_duration = final_duration

    out: dict[str, np.ndarray | pd.Series] = {
        "Id": numeric_df["Id"].to_numpy(),
        "start_time": start_time,
        "end_time": observed_end_time,
        "process_duration": observed_duration,
    }
    if "Response" in numeric_df:
        out["Response"] = numeric_df["Response"].astype("int8").to_numpy()

    all_stations = sorted(set(numeric_groups) | set(date_groups))
    line_counts: dict[int, np.ndarray] = {}
    total_numeric_observed = np.zeros(n_rows, dtype=np.int32)
    total_numeric_possible = 0
    prefix_station_count = np.zeros(n_rows, dtype=np.int16)

    for station in all_stations:
        label = station.label
        station_time = station_times.get(station, np.full(n_rows, np.nan))
        date_seen = np.isfinite(station_time)
        numeric_columns = numeric_groups.get(station, [])
        if numeric_columns:
            values = numeric_df[numeric_columns].to_numpy(dtype=float, copy=False)
            finite_count = np.isfinite(values).sum(axis=1).astype(np.int16)
            numeric_seen = finite_count > 0
            total_numeric_possible += len(numeric_columns)
        else:
            values = None
            finite_count = np.zeros(n_rows, dtype=np.int16)
            numeric_seen = np.zeros(n_rows, dtype=bool)

        if prefix_fraction >= 1.0:
            active = date_seen | numeric_seen
        else:
            active = date_seen & np.isfinite(cutoff) & (station_time <= cutoff)
        prefix_station_count += active.astype(np.int16)
        line_counts.setdefault(station.line, np.zeros(n_rows, dtype=np.int16))
        line_counts[station.line] += active.astype(np.int16)

        if options.station_presence:
            out[f"{label}__seen"] = active.astype(np.int8)
        if options.station_time:
            rel = station_time - start_time
            rel[~active] = np.nan
            out[f"{label}__time_from_start"] = rel

        if not numeric_columns or values is None:
            continue

        mean = _row_nanmean(values)
        std = _row_nanstd(values)
        mean[~active] = np.nan
        std[~active] = np.nan
        visible_count = np.where(active, finite_count, 0).astype(np.int16)
        total_numeric_observed += visible_count

        if options.numeric_mean:
            out[f"{label}__num_mean"] = mean
        if options.numeric_std:
            out[f"{label}__num_std"] = std
        if options.numeric_count:
            out[f"{label}__num_count"] = visible_count

    out["prefix_station_count"] = prefix_station_count
    out["numeric_observed_count"] = total_numeric_observed
    if total_numeric_possible:
        out["numeric_missing_ratio"] = 1.0 - (total_numeric_observed / float(total_numeric_possible))
    else:
        out["numeric_missing_ratio"] = np.nan

    if options.line_station_counts:
        for line, counts in sorted(line_counts.items()):
            out[f"L{line}__station_count"] = counts

    return pd.DataFrame(out)
