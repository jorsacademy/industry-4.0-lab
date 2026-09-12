import pandas as pd

from src.train import chronological_blocks


def test_chronological_blocks_apply_time_purge():
    ts = pd.Series(pd.date_range("2024-01-01", periods=100, freq="2min"))
    active = pd.Series([True] * 100)
    blocks, bounds = chronological_blocks(
        ts,
        active,
        fit_fraction=0.55,
        selection_fraction=0.15,
        calibration_fraction=0.15,
        purge_minutes=10,
    )
    assert ts.iloc[blocks["fit"][-1]] <= bounds["selection_boundary"] - pd.Timedelta(minutes=10)
    assert ts.iloc[blocks["selection"][0]] >= bounds["selection_boundary"] + pd.Timedelta(minutes=10)
    assert ts.iloc[blocks["calibration"][0]] >= bounds["calibration_boundary"] + pd.Timedelta(minutes=10)
    assert ts.iloc[blocks["test"][0]] >= bounds["test_boundary"] + pd.Timedelta(minutes=10)
