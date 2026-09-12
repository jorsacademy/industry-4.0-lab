import numpy as np

from src.modeling import choose_threshold, classification_metrics


def test_threshold_respects_minimum_recall_when_feasible():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    p = np.array([0.05, 0.10, 0.20, 0.40, 0.30, 0.60, 0.80, 0.90])
    result = choose_threshold(y, p, min_recall=0.75)
    assert result.recall >= 0.75
    metrics = classification_metrics(y, p, result.threshold)
    assert metrics["recall"] >= 0.75
    assert 0 <= metrics["average_precision"] <= 1
