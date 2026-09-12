import numpy as np
import pandas as pd

from src.modeling import chronological_blocks, conformal_radius, regression_metrics


def test_chronological_blocks_preserve_order():
    frame = pd.DataFrame({
        "HEATID": [f"H{i}" for i in range(100)],
        "target_time": pd.date_range("2024-01-01", periods=100, freq="h"),
        "target_temp": np.linspace(1500, 1600, 100),
    })
    blocks = chronological_blocks(frame, (0.55, 0.15, 0.15, 0.15))
    assert blocks["fit"]["target_time"].max() < blocks["selection"]["target_time"].min()
    assert blocks["selection"]["target_time"].max() < blocks["calibration"]["target_time"].min()
    assert blocks["calibration"]["target_time"].max() < blocks["test"]["target_time"].min()


def test_metrics_and_conformal_radius_are_finite():
    y = np.array([1500.0, 1510.0, 1520.0, 1530.0])
    p = np.array([1502.0, 1507.0, 1524.0, 1528.0])
    metrics = regression_metrics(y, p)
    assert metrics["mae"] > 0
    assert np.isfinite(metrics["rmse"])
    assert conformal_radius(y, p, 0.75) >= 2.0
