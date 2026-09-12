import pandas as pd

from src.monitoring import summarize_lots, wilson_interval


def test_lot_summary_flags_only_stable_high_rate_lots():
    df = pd.DataFrame({
        "LOT": ["A"] * 4 + ["B"] * 2,
        "is_defect": [0, 0, 0, 1, 0, 1],
        "timestamp": pd.date_range("2020-01-01", periods=6, freq="s"),
    })
    out = summarize_lots(df, min_lot_size=3, warning_rate=0.20).set_index("LOT")
    assert bool(out.loc["A", "warning"])
    assert not bool(out.loc["B", "warning"])


def test_wilson_interval_bounds_probability():
    low, high = wilson_interval(5, 100)
    assert 0 <= low < 0.05 < high <= 1
