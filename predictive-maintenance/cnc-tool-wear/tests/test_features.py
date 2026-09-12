import numpy as np
import pandas as pd

from src.features import add_tracking_errors, extract_window_features, make_windows


def synthetic_experiment(rows: int = 60) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "X1_ActualPosition": x + 0.5,
            "X1_CommandPosition": x,
            "X1_ActualVelocity": np.ones(rows) * 2.0,
            "X1_CommandVelocity": np.ones(rows) * 1.8,
            "X1_CurrentFeedback": np.linspace(1.0, 2.0, rows),
            "S1_OutputPower": np.linspace(0.1, 0.3, rows),
            "M1_CURRENT_FEEDRATE": np.ones(rows) * 6,
            "Machining_Process": ["Layer 1 Up"] * rows,
            "_known_artifact": np.zeros(rows),
            "experiment_id": [1] * rows,
            "tool_condition": ["worn"] * rows,
        }
    )


def test_tracking_error_is_actual_minus_commanded():
    frame = add_tracking_errors(synthetic_experiment())
    assert np.allclose(frame["X1_PositionTrackingError"], 0.5)
    assert np.allclose(frame["X1_VelocityTrackingError"], 0.2)


def test_extract_window_features_contains_signal_statistics():
    features = extract_window_features(
        synthetic_experiment(50),
        statistics=["mean", "std", "rms", "slope"],
        sample_period_seconds=0.1,
    )
    assert "S1_OutputPower__mean" in features
    assert "X1_PositionTrackingError__mean" in features
    assert features["machining_process"] == "Layer 1 Up"
    assert features["window_duration_seconds"] == 5.0


def test_make_windows_preserves_group_and_target():
    table = make_windows(
        synthetic_experiment(100),
        window_size=50,
        step_size=25,
        minimum_rows_per_window=40,
        statistics=["mean", "std"],
        sample_period_seconds=0.1,
        target="tool_condition",
    )
    assert set(table["experiment_id"]) == {1}
    assert set(table["target"]) == {"worn"}
    assert len(table) >= 3
