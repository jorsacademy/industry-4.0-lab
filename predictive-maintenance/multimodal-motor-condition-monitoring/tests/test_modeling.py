import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import blocked_purged_folds
from src.modeling import evaluate_candidate


def test_candidate_evaluation_returns_complete_oof_predictions():
    rows = []
    for label, offset in [("a", -3.0), ("b", 3.0)]:
        for i in range(30):
            rows.append({"condition": label, "window_index": i, "x": offset + i * 0.001})
    df = pd.DataFrame(rows)
    folds = blocked_purged_folds(df, n_splits=3, purge_windows=1)
    metrics, pred = evaluate_candidate(df[["x"]], df["condition"], folds, LogisticRegression())
    assert len(pred) == len(df)
    assert metrics["macro_f1"] > 0.9
