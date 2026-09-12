import pandas as pd

from src.data import build_snapshot_dataset


def _table(rows):
    frame = pd.DataFrame(rows)
    frame["_TS"] = pd.to_datetime(frame["_TS"])
    return frame


def test_snapshot_excludes_future_process_events():
    tables = {
        "temperature": _table([
            {"HEATID": "H1", "_TS": "2024-01-01 10:00", "TEMP": 1500, "VALO2_PPM": 100},
            {"HEATID": "H1", "_TS": "2024-01-01 10:10", "TEMP": 1560, "VALO2_PPM": 120},
        ]),
        "transformer": _table([
            {"HEATID": "H1", "_TS": "2024-01-01 09:58", "MW": 10, "DURATION": 30, "TAP": 4},
            {"HEATID": "H1", "_TS": "2024-01-01 10:05", "MW": 99, "DURATION": 30, "TAP": 9},
        ]),
        "basket": _table([
            {"HEATID": "H1", "_TS": "2024-01-01 09:30", "CHARGE_AMOUNT": 100, "MAT_CODE": "A"},
        ]),
        "added": _table([
            {"HEATID": "H1", "_TS": "2024-01-01 10:05", "CHARGE_AMOUNT": 50, "MAT_CODE": "B"},
        ]),
    }
    row = build_snapshot_dataset(tables).iloc[0]
    assert row["target_temp"] == 1560
    assert row["snapshot_temp"] == 1500
    assert row["transformer_mw_sum"] == 10
    assert row["basket_charge_total"] == 100
    assert row["added_charge_total"] == 0
