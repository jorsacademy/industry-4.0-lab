import numpy as np
import pandas as pd

from src.data import build_supervised, chronological_blocks


def _synthetic_frame(n=20):
    data = {
        "time_stamp": pd.date_range("2024-01-01", periods=n, freq="s"),
        "Machine1.MotorRPM.C.Actual": np.arange(n, dtype=float),
    }
    for stage in [1, 2]:
        for i in range(15):
            data[f"Stage{stage}.Output.Measurement{i}.U.Actual"] = np.arange(n, dtype=float) + 100 * stage + i
            data[f"Stage{stage}.Output.Measurement{i}.U.Setpoint"] = np.full(n, 10.0 + i)
    return pd.DataFrame(data)


def test_supervised_alignment_uses_future_target_but_current_features():
    frame = _synthetic_frame()
    data = build_supervised(frame, horizon_rows=3)
    assert len(data.X) == len(frame) - 3
    assert data.X.iloc[0]["Machine1.MotorRPM.C.Actual"] == 0.0
    assert data.y.iloc[0]["Stage2.Output.Measurement0.U.Actual"] == frame.iloc[3]["Stage2.Output.Measurement0.U.Actual"]
    assert data.persistence.iloc[0]["Stage2.Output.Measurement0.U.Actual"] == frame.iloc[0]["Stage2.Output.Measurement0.U.Actual"]
    assert "Stage2.Output.Measurement0.U.Actual" not in data.feature_columns


def test_nonpositive_future_setpoint_marks_target_invalid():
    frame = _synthetic_frame()
    frame.loc[5, "Stage2.Output.Measurement7.U.Setpoint"] = 0.0
    data = build_supervised(frame, horizon_rows=2)
    assert not bool(data.valid_target.iloc[3])


def test_chronological_blocks_are_ordered_and_purged():
    blocks = chronological_blocks(1000, 0.55, 0.15, 0.15, 0.15, purge_rows=10)
    assert blocks["fit"][-1] < blocks["selection"][0]
    assert blocks["selection"][-1] < blocks["calibration"][0]
    assert blocks["calibration"][-1] < blocks["test"][0]
    assert blocks["selection"][0] - blocks["fit"][-1] >= 20
