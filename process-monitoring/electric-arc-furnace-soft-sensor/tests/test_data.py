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
        "gas": _table([
            {"HEATID": "H1", "_TS": "2024-01-01 09:59", "O2_AMOUNT": 2, "GAS_AMOUNT": 1, "O2_FLOW": 3, "GAS_FLOW": 4},
            {"HEATID": "H1", "_TS": "2024-01-01 10:05", "O2_AMOUNT": 20, "GAS_AMOUNT": 10, "O2_FLOW": 30, "GAS_FLOW": 40},
        ]),
        "carbon": _table([
            {"HEATID": "H1", "_TS": "2024-01-01 09:59", "INJ_AMOUNT_CARBON": 1, "INJ_FLOW_CARBON": 2},
            {"HEATID": "H1", "_TS": "2024-01-01 10:05", "INJ_AMOUNT_CARBON": 9, "INJ_FLOW_CARBON": 8},
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
    assert row["o2_amount_last"] == 2
    assert row["carbon_amount_last"] == 1
    assert row["basket_charge_total"] == 100
    assert row["added_charge_total"] == 0
