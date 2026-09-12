from pathlib import Path

import numpy as np

from src.data import chronological_blocks, feature_columns, load_secom


def test_loader_parses_labels_and_timestamp(tmp_path: Path):
    data = tmp_path / "secom.data"
    labels = tmp_path / "secom_labels.data"
    data.write_text("1 NaN 3\n4 5 6\n", encoding="utf-8")
    labels.write_text("-1 19/07/2008 11:55:00\n1 19/07/2008 12:00:00\n", encoding="utf-8")
    df = load_secom(data, labels)
    assert feature_columns(df) == ["V000", "V001", "V002"]
    assert df["is_fail"].tolist() == [0, 1]
    assert np.isnan(df.loc[0, "V001"])
    assert df["timestamp"].notna().all()


def test_chronological_blocks_have_no_time_overlap():
    import pandas as pd
    rows = []
    for i in range(40):
        rows.append({
            "V000": float(i), "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i),
            "raw_label": 1 if i % 5 == 0 else -1, "is_fail": 1 if i % 5 == 0 else 0, "source_row": i,
        })
    df = pd.DataFrame(rows)
    blocks = chronological_blocks(df, fit_fraction=0.50, selection_fraction=0.20, calibration_fraction=0.15, test_fraction=0.15)
    assert blocks.fit["timestamp"].max() < blocks.selection["timestamp"].min()
    assert blocks.selection["timestamp"].max() < blocks.calibration["timestamp"].min()
    assert blocks.calibration["timestamp"].max() < blocks.test["timestamp"].min()
