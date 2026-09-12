import numpy as np
import pandas as pd

from src.features import extract_station_features


def fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = pd.DataFrame(
        {
            "Id": [1, 2],
            "L0_S0_F0": [1.0, np.nan],
            "L0_S0_F1": [3.0, np.nan],
            "L0_S1_F2": [10.0, 20.0],
            "Response": [0, 1],
        }
    )
    dates = pd.DataFrame(
        {
            "Id": [1, 2],
            "L0_S0_D0": [0.0, np.nan],
            "L0_S1_D2": [10.0, 4.0],
        }
    )
    return numeric, dates


def test_full_features_preserve_station_aggregates() -> None:
    numeric, dates = fixture()
    out = extract_station_features(numeric, dates, prefix_fraction=1.0)
    assert out.loc[0, "L0_S0__num_mean"] == 2.0
    assert out.loc[0, "L0_S1__num_mean"] == 10.0
    assert out.loc[1, "L0_S0__seen"] == 0
    assert out.loc[1, "L0_S1__seen"] == 1
    assert out.loc[0, "process_duration"] == 10.0


def test_early_prefix_masks_future_station() -> None:
    numeric, dates = fixture()
    out = extract_station_features(numeric, dates, prefix_fraction=0.25)
    assert out.loc[0, "L0_S0__seen"] == 1
    assert out.loc[0, "L0_S1__seen"] == 0
    assert np.isnan(out.loc[0, "L0_S1__num_mean"])
    assert out.loc[0, "L0_S1__num_count"] == 0
    assert out.loc[0, "numeric_observed_count"] == 2
    assert out.loc[0, "end_time"] == 2.5
    assert out.loc[0, "process_duration"] == 2.5
