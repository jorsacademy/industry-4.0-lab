import numpy as np

from src.features import FeatureConfig, extract_window_features


def test_order_features_follow_shaft_frequency():
    fs = 4096
    seconds = 4
    n = fs * seconds
    t = np.arange(n) / fs
    rpm = np.full(n, 1800.0)  # 30 Hz shaft order
    signal = 2.0 * np.sin(2 * np.pi * 30 * t) + 0.1 * np.sin(2 * np.pi * 60 * t)
    window = np.column_stack([
        np.full(n, 4.0),
        rpm,
        signal,
        0.8 * signal,
        0.5 * signal,
    ])
    columns = ["V_in", "Measured_RPM", "Vibration_1", "Vibration_2", "Vibration_3"]
    features = extract_window_features(window, columns, FeatureConfig(sample_rate_hz=fs))
    assert features["compact__s1__order1_amp"] > 5 * features["compact__s1__order2_amp"]
    assert features["compact__rpm_median"] == 1800.0
    assert features["order__s1__o1p00"] > features["order__s1__o2p00"]
