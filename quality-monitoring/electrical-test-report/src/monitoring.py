from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import load_electrical_report, resolve_column


def wilson_interval(defects: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = defects / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_lots(
    df: pd.DataFrame,
    *,
    min_lot_size: int = 100,
    warning_rate: float = 0.05,
) -> pd.DataFrame:
    group_col = resolve_column(df.columns, "LOT", aliases=("lot no", "lot number", "batch", "batch id"))
    if "is_defect" not in df.columns:
        raise KeyError("Expected an 'is_defect' column. Load data with load_electrical_report().")

    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            observations=("is_defect", "size"),
            defects=("is_defect", "sum"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    summary["defect_rate"] = summary["defects"] / summary["observations"]
    intervals = [wilson_interval(int(d), int(n)) for d, n in zip(summary["defects"], summary["observations"])]
    summary["ci95_low"] = [x[0] for x in intervals]
    summary["ci95_high"] = [x[1] for x in intervals]
    summary["stable_lot"] = summary["observations"].ge(min_lot_size)
    summary["warning"] = summary["stable_lot"] & summary["defect_rate"].ge(warning_rate)
    return summary.sort_values(["warning", "defect_rate", "observations"], ascending=[False, False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lot-level electrical-test quality monitoring summary.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--min-lot-size", type=int, default=100)
    parser.add_argument("--warning-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("reports/lot_quality_summary.csv"))
    args = parser.parse_args()

    df = load_electrical_report(args.csv)
    report = summarize_lots(df, min_lot_size=args.min_lot_size, warning_rate=args.warning_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(report.head(20).to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
