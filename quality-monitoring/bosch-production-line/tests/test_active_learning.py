import numpy as np
import pandas as pd

from src.active_learning import benchmark_active_learning_strategies, top_fraction_capture


def _fixture() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 180
    y = (np.arange(n) % 7 == 0).astype(int)
    return pd.DataFrame(
        {
            "Id": np.arange(n),
            "start_time": np.arange(n, dtype=float),
            "end_time": np.arange(n, dtype=float) + 1.0,
            "Response": y,
            "f1": y + rng.normal(0.0, 0.7, n),
            "f2": rng.normal(0.0, 1.0, n),
            "f3": 0.5 * y + rng.normal(0.0, 1.0, n),
        }
    )


def test_top_fraction_capture_is_bounded() -> None:
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.4])
    value = top_fraction_capture(y, p, inspection_fraction=0.5)
    assert 0.0 <= value <= 1.0


def test_all_active_learning_strategies_produce_monotone_label_budgets() -> None:
    frame = _fixture()
    result = benchmark_active_learning_strategies(
        frame,
        np.arange(120),
        np.arange(120, 180),
        initial_labels=20,
        query_batch_size=10,
        rounds=2,
        candidate_pool_size=80,
        random_state=11,
    )
    assert set(result["strategy"]) == {"random", "uncertainty", "diversity", "hybrid"}
    for _, group in result.groupby("strategy"):
        assert group["labels_used"].is_monotonic_increasing
        assert len(group) == 3
        assert group["average_precision"].between(0.0, 1.0).all()
