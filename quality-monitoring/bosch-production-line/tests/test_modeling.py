import numpy as np
import pandas as pd

from src.modeling import select_operating_threshold, temporal_four_way_split


def test_temporal_split_has_strictly_ordered_boundaries() -> None:
    frame = pd.DataFrame({"start_time": np.arange(100, dtype=float)})
    split = temporal_four_way_split(
        frame,
        train_fraction=0.60,
        selection_fraction=0.15,
        calibration_fraction=0.10,
        test_fraction=0.15,
    )
    assert frame.iloc[split.train]["start_time"].max() < frame.iloc[split.selection]["start_time"].min()
    assert frame.iloc[split.selection]["start_time"].max() < frame.iloc[split.calibration]["start_time"].min()
    assert frame.iloc[split.calibration]["start_time"].max() < frame.iloc[split.test]["start_time"].min()


def test_threshold_respects_precision_floor_when_possible() -> None:
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.70, 0.90])
    threshold = select_operating_threshold(y, p, strategy="min_precision", min_precision=0.80)
    predicted = p >= threshold
    precision = y[predicted].mean()
    assert precision >= 0.80
