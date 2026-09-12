from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import grouped_oof, regression_metrics


def test_grouped_oof_covers_all_runs():
    rows = []
    for condition in range(10):
        for replica in range(2):
            rows.append({
                "x": float(condition),
                "y": 2.0 * condition + replica * 0.01,
                "condition": f"C{condition}",
            })
    frame = pd.DataFrame(rows)
    model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pred = grouped_oof(model, frame[["x"]], frame["y"], frame["condition"], n_splits=5)
    assert np.isfinite(pred).all()
    assert len(pred) == len(frame)
    metrics = regression_metrics(frame["y"].to_numpy(), pred)
    assert set(metrics) == {"mae", "rmse", "r2"}


def test_grouped_oof_rejects_single_condition():
    frame = pd.DataFrame({"x": [1.0, 1.0], "y": [1.0, 1.1], "g": ["A", "A"]})
    model = Pipeline([("model", Ridge())])
    try:
        grouped_oof(model, frame[["x"]], frame["y"], frame["g"], n_splits=5)
    except ValueError as exc:
        assert "two unique condition groups" in str(exc)
    else:
        raise AssertionError("Expected grouped_oof to reject one group")
