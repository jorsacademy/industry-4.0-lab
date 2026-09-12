import numpy as np
import pandas as pd

from src.data import compact_feature_columns, order_feature_columns
from src.modeling import blocked_splits, expected_calibration_error


def test_blocked_splits_hold_out_exact_block():
    frame = pd.DataFrame({"dev_block": np.tile(np.arange(5), 4)})
    splits = blocked_splits(frame, n_folds=5)
    for fold, (train, test) in enumerate(splits):
        assert set(frame.iloc[test]["dev_block"]) == {fold}
        assert fold not in set(frame.iloc[train]["dev_block"])


def test_sensor_feature_selection():
    frame = pd.DataFrame(columns=[
        "compact__rpm_median",
        "compact__s1__rms",
        "compact__s2__rms",
        "compact__s3__rms",
        "order__rpm_median",
        "order__s1__o1p00",
        "order__s2__o1p00",
        "order__s3__o1p00",
    ])
    assert compact_feature_columns(frame, sensors=(2,)) == ["compact__rpm_median", "compact__s2__rms"]
    assert order_feature_columns(frame, sensors=(1, 3)) == [
        "order__rpm_median", "order__s1__o1p00", "order__s3__o1p00"
    ]


def test_perfect_confidence_has_zero_ece():
    y = np.array([0, 1, 2])
    probs = np.eye(3)
    assert expected_calibration_error(y, probs, n_bins=5) == 0.0
