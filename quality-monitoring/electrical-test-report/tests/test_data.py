from pathlib import Path

import pandas as pd

from src.data import load_electrical_report, measurement_columns


def test_utf16_and_repeated_header_cleanup(tmp_path: Path):
    cols = ["Time", "LOT", "Result", *[f"F{i}" for i in range(1, 20)]]
    valid1 = ["2019-10-04 10:00:00", "A", "1", *range(1, 20)]
    repeated = cols.copy()
    valid2 = ["2019-10-04 10:00:01", "B", "0", *range(2, 21)]
    df = pd.DataFrame([valid1, repeated, valid2], columns=cols)
    path = tmp_path / "Electrical_Test_Report.csv"
    df.to_csv(path, index=False, encoding="utf-16")

    clean = load_electrical_report(path)
    assert len(clean) == 2
    assert clean["is_defect"].tolist() == [False, True]
    assert len(measurement_columns(clean)) == 19
    assert clean["timestamp"].notna().all()


def test_measurement_sort_is_numeric():
    df = pd.DataFrame(columns=["F10", "F2", "F1", "Other"])
    assert measurement_columns(df) == ["F1", "F2", "F10"]
