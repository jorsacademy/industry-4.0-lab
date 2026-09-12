import numpy as np

from src.modeling import block_bootstrap_improvement_ci, conformal_radius, target_scale


def test_target_scale_falls_back_when_iqr_is_zero():
    y = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    scale = target_scale(y)
    assert np.all(scale > 0)


def test_conformal_radius_covers_calibration_residual_quantile():
    y = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    pred = np.array([[0.0], [1.0], [1.0], [2.0], [2.0]])
    radius = conformal_radius(y, pred, coverage=0.8)
    assert radius.shape == (1,)
    assert radius[0] >= 1.0


def test_block_bootstrap_positive_improvement_when_model_error_is_lower():
    model = np.full(300, 0.5)
    baseline = np.full(300, 0.8)
    result = block_bootstrap_improvement_ci(model, baseline, block_length=20, repetitions=100, seed=42)
    assert result["mean_improvement"] > 0
    assert result["ci95_low"] > 0
