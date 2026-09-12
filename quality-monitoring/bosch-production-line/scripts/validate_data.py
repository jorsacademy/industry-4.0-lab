from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import yaml

NUMERIC_RE = re.compile(r"^L\d+_S\d+_F\d+$")
DATE_RE = re.compile(r"^L\d+_S\d+_D\d+$")


def count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--full", action="store_true", help="Count all rows in train_numeric.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    numeric_path = raw_dir / data_cfg["train_numeric"]
    date_path = raw_dir / data_cfg["train_date"]
    if not numeric_path.exists() or not date_path.exists():
        raise FileNotFoundError("Missing train_numeric.csv or train_date.csv. Run scripts/download_data.py first.")

    numeric_header = header(numeric_path)
    date_header = header(date_path)
    numeric_features = [c for c in numeric_header if NUMERIC_RE.match(c)]
    date_features = [c for c in date_header if DATE_RE.match(c)]
    assert numeric_header[0] == "Id"
    assert "Response" in numeric_header
    assert date_header[0] == "Id"
    assert len(numeric_features) == 968, f"Expected 968 numeric features, found {len(numeric_features)}"
    assert len(date_features) == 1156, f"Expected 1156 date features, found {len(date_features)}"

    sample_numeric = pd.read_csv(numeric_path, usecols=["Id", "Response"], nrows=10000)
    sample_date = pd.read_csv(date_path, usecols=["Id"], nrows=10000)
    assert sample_numeric["Id"].equals(sample_date["Id"]), "First 10k numeric/date Ids are not aligned"
    assert set(sample_numeric["Response"].dropna().unique()).issubset({0, 1})

    result = {
        "numeric_features": len(numeric_features),
        "date_features": len(date_features),
        "sample_rows_checked": len(sample_numeric),
        "sample_positive_rate": float(sample_numeric["Response"].mean()),
    }
    if args.full:
        rows = count_rows(numeric_path)
        expected = int(data_cfg.get("expected_train_rows", 1183747))
        assert rows == expected, f"Expected {expected} rows, found {rows}"
        result["train_rows"] = rows
    print(result)


if __name__ == "__main__":
    main()
