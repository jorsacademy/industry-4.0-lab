import numpy as np
import pandas as pd

from src.few_label import few_label_adaptation_curve, sample_k_per_class


def _fixture() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n = 210
    y = (np.arange(n) % 5 == 0).astype(int)
    return pd.DataFrame(
        {
            "Id": np.arange(n),
            "start_time": np.arange(n, dtype=float),
            "end_time": np.arange(n, dtype=float) + 1.0,
            "Response": y,
            "f1": 1.2 * y + rng.normal(0.0, 0.8, n),
            "f2": rng.normal(0.0, 1.0, n),
        }
    )


def test_k_per_class_sampler_returns_balanced_shots() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    idx = sample_k_per_class(y, 2, np.random.default_rng(1))
    assert len(idx) == 4
    assert np.bincount(y[idx], minlength=2).tolist() == [2, 2]


def test_few_label_curve_keeps_test_partition_untouched() -> None:
    frame = _fixture()
    result = few_label_adaptation_curve(
        frame,
        np.arange(100),
        np.arange(100, 160),
        np.arange(160, 210),
        k_values=(1, 2),
        repeats=2,
        target_weight=4.0,
        random_state=5,
    )
    assert set(result["method"]) == {"source_only", "target_only", "source_plus_few_labels"}
    adapted = result[result["method"] == "source_plus_few_labels"]
    assert set(adapted["target_labels"]) == {2, 4}
    assert result["average_precision"].between(0.0, 1.0).all()
