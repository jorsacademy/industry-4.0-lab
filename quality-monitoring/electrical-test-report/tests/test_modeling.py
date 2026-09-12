import numpy as np
import pandas as pd

from src.modeling import build_candidates, choose_threshold_for_recall, out_of_fold_probabilities


def test_grouped_oof_covers_all_rows():
    rng = np.random.default_rng(0)
    groups = pd.Series(np.repeat([f"L{i}" for i in range(10)], 10))
    y = pd.Series(np.tile([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 10))
    X = pd.DataFrame({"F1": rng.normal(size=100), "F2": rng.normal(size=100)})
    model = build_candidates(1)["logistic_regression"]
    prob, fold = out_of_fold_probabilities(model, X, y, groups, n_splits=5, random_state=1)
    assert np.isfinite(prob).all()
    assert set(fold) == {0, 1, 2, 3, 4}


def test_threshold_targets_recall():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.9])
    threshold = choose_threshold_for_recall(y, p, target_recall=1.0)
    assert threshold <= 0.6
