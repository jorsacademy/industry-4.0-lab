import numpy as np
import pandas as pd

from src.features import drift_report, fit_feature_screen, panel_features


def test_screen_is_fit_only_and_drops_bad_columns():
    df = pd.DataFrame({
        "V000": [0, 1, 0, 1, 0, 1],
        "V001": [1, 1, 1, 1, 1, 1],
        "V002": [np.nan, np.nan, np.nan, np.nan, 2, 3],
        "V003": [1, 2, 3, 4, 5, 6],
        "is_fail": [0, 1, 0, 1, 0, 1],
    })
    screen = fit_feature_screen(df, ["V000", "V001", "V002", "V003"], max_missing_rate=0.50)
    assert "V001" not in screen.eligible
    assert "V002" not in screen.eligible
    assert set(panel_features(screen, 1)).issubset(set(screen.eligible))
    assert len(panel_features(screen, "all")) == len(screen.eligible)


def test_drift_report_orders_large_shift_first():
    history = pd.DataFrame({"V000": [0, 1, 2, 3], "V001": [10, 11, 12, 13]})
    future = pd.DataFrame({"V000": [20, 21, 22, 23], "V001": [10, 11, 12, 13]})
    out = drift_report(history, future, ["V000", "V001"])
    assert out.iloc[0]["feature"] == "V000"
