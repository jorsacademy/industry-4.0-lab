from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt", class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
    }


def evaluate_candidate(X: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]], estimator: object) -> tuple[dict[str, float], np.ndarray]:
    pred = np.empty(len(y), dtype=object)
    seen = np.zeros(len(y), dtype=bool)
    fold_metrics: list[tuple[float, float, float]] = []
    for train_idx, test_idx in folds:
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = model.predict(X.iloc[test_idx])
        pred[test_idx] = p
        seen[test_idx] = True
        fold_metrics.append((
            f1_score(y.iloc[test_idx], p, average="macro"),
            balanced_accuracy_score(y.iloc[test_idx], p),
            accuracy_score(y.iloc[test_idx], p),
        ))
    if not seen.all():
        raise RuntimeError("Blocked CV did not produce a prediction for every row.")
    scores = np.asarray(fold_metrics, dtype=float)
    return {
        "macro_f1": float(scores[:, 0].mean()),
        "balanced_accuracy": float(scores[:, 1].mean()),
        "accuracy": float(scores[:, 2].mean()),
        "macro_f1_std": float(scores[:, 0].std(ddof=1)) if len(scores) > 1 else 0.0,
    }, pred
