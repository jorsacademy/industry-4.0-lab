from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import feature_columns


def make_few_label_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def sample_k_per_class(
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if k <= 0:
        raise ValueError("k must be positive")
    selected: list[int] = []
    for label in sorted(np.unique(y)):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < k:
            raise ValueError(f"class {label} has fewer than {k} examples")
        selected.extend(map(int, rng.choice(candidates, size=k, replace=False)))
    return np.asarray(sorted(selected), dtype=int)


def _evaluate(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(x)[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y, probabilities)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


def few_label_adaptation_curve(
    frame: pd.DataFrame,
    source_idx: np.ndarray,
    adaptation_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    k_values: Iterable[int] = (1, 2, 5, 10, 20, 50),
    repeats: int = 5,
    target_weight: float = 8.0,
    random_state: int = 42,
) -> pd.DataFrame:
    columns = feature_columns(frame)
    x = frame[columns]
    y = frame["Response"].astype(int).to_numpy()
    source_idx = np.asarray(source_idx, dtype=int)
    adaptation_idx = np.asarray(adaptation_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if target_weight <= 0:
        raise ValueError("target_weight must be positive")
    for name, idx in (
        ("source", source_idx),
        ("adaptation", adaptation_idx),
        ("test", test_idx),
    ):
        if np.unique(y[idx]).size != 2:
            raise ValueError(f"{name} partition must contain both classes")

    rows: list[dict[str, float | int | str]] = []
    source_model = make_few_label_model(random_state)
    source_model.fit(x.iloc[source_idx], y[source_idx])
    rows.append(
        {
            "method": "source_only",
            "k_per_class": 0,
            "repeat": 0,
            "target_labels": 0,
            **_evaluate(source_model, x.iloc[test_idx], y[test_idx]),
        }
    )

    rng = np.random.default_rng(random_state)
    adaptation_y = y[adaptation_idx]
    for k in map(int, k_values):
        for repeat in range(repeats):
            local = sample_k_per_class(adaptation_y, k, rng)
            target_idx = adaptation_idx[local]

            target_only = make_few_label_model(random_state + 1000 + 100 * k + repeat)
            target_only.fit(x.iloc[target_idx], y[target_idx])
            rows.append(
                {
                    "method": "target_only",
                    "k_per_class": k,
                    "repeat": repeat,
                    "target_labels": int(len(target_idx)),
                    **_evaluate(target_only, x.iloc[test_idx], y[test_idx]),
                }
            )

            combined_idx = np.concatenate([source_idx, target_idx])
            sample_weight = np.concatenate(
                [
                    np.ones(len(source_idx), dtype=float),
                    np.full(len(target_idx), target_weight, dtype=float),
                ]
            )
            adapted = make_few_label_model(random_state + 2000 + 100 * k + repeat)
            adapted.fit(
                x.iloc[combined_idx],
                y[combined_idx],
                model__sample_weight=sample_weight,
            )
            rows.append(
                {
                    "method": "source_plus_few_labels",
                    "k_per_class": k,
                    "repeat": repeat,
                    "target_labels": int(len(target_idx)),
                    **_evaluate(adapted, x.iloc[test_idx], y[test_idx]),
                }
            )

    return pd.DataFrame(rows)
