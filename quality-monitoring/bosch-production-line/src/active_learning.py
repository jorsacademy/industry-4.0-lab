from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import feature_columns


def make_active_learning_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def top_fraction_capture(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    inspection_fraction: float = 0.01,
) -> float:
    if not 0 < inspection_fraction <= 1:
        raise ValueError("inspection_fraction must be in (0, 1]")
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y_true) * inspection_fraction)))
    top = np.argpartition(probabilities, -k)[-k:]
    return float(y_true[top].sum() / positives)


def _binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (probabilities >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
    }


def _initial_label_indices(
    y_pool: np.ndarray,
    initial_labels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y_pool = np.asarray(y_pool, dtype=int)
    if initial_labels < 2 or initial_labels > len(y_pool):
        raise ValueError("initial_labels must be in [2, pool size]")
    order = rng.permutation(len(y_pool))
    chosen = list(map(int, order[:initial_labels]))
    cursor = initial_labels
    while np.unique(y_pool[chosen]).size < 2 and cursor < len(y_pool):
        chosen.append(int(order[cursor]))
        cursor += 1
    if np.unique(y_pool[chosen]).size < 2:
        raise ValueError("active-learning pool must contain both classes")
    return np.asarray(chosen, dtype=int)


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = mad if mad > 1e-12 else float(np.std(values) + 1e-12)
    return (values - median) / scale


def _query_batch(
    model: Pipeline,
    x_pool: pd.DataFrame,
    unlabeled_idx: np.ndarray,
    labeled_idx: np.ndarray,
    *,
    batch_size: int,
    strategy: str,
    rng: np.random.Generator,
    candidate_pool_size: int,
) -> np.ndarray:
    if len(unlabeled_idx) == 0:
        return np.array([], dtype=int)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    if len(unlabeled_idx) > candidate_pool_size:
        candidates = rng.choice(unlabeled_idx, size=candidate_pool_size, replace=False)
    else:
        candidates = np.asarray(unlabeled_idx, dtype=int)
    k = min(batch_size, len(candidates))

    if strategy == "random":
        return np.asarray(rng.choice(candidates, size=k, replace=False), dtype=int)

    probabilities = model.predict_proba(x_pool.iloc[candidates])[:, 1]
    uncertainty = 1.0 - np.abs(probabilities - 0.5) / 0.5
    if strategy == "uncertainty":
        return candidates[np.argsort(uncertainty)[-k:]]

    if strategy not in {"diversity", "hybrid"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    labeled_sample = np.asarray(labeled_idx, dtype=int)
    if len(labeled_sample) > candidate_pool_size:
        labeled_sample = rng.choice(labeled_sample, size=candidate_pool_size, replace=False)

    union = np.concatenate([candidates, labeled_sample])
    transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    embedding = transformer.fit_transform(x_pool.iloc[union])
    candidate_embedding = embedding[: len(candidates)]
    labeled_embedding = embedding[len(candidates) :]
    center = (
        labeled_embedding.mean(axis=0)
        if len(labeled_embedding)
        else np.zeros(candidate_embedding.shape[1], dtype=float)
    )
    diversity = np.linalg.norm(candidate_embedding - center, axis=1)

    score = diversity if strategy == "diversity" else _robust_z(uncertainty) + _robust_z(diversity)
    return candidates[np.argsort(score)[-k:]]


def active_learning_curve(
    frame: pd.DataFrame,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    strategy: str = "uncertainty",
    initial_labels: int = 2000,
    query_batch_size: int = 500,
    rounds: int = 6,
    random_state: int = 42,
    candidate_pool_size: int = 5000,
    inspection_fraction: float = 0.01,
) -> pd.DataFrame:
    columns = feature_columns(frame)
    x = frame[columns]
    y = frame["Response"].astype(int).to_numpy()
    pool_idx = np.asarray(pool_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    if np.intersect1d(pool_idx, test_idx).size:
        raise ValueError("pool_idx and test_idx must not overlap")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")

    x_pool = x.iloc[pool_idx].reset_index(drop=True)
    y_pool = y[pool_idx]
    rng = np.random.default_rng(random_state)
    labeled = set(map(int, _initial_label_indices(y_pool, min(initial_labels, len(pool_idx)), rng)))

    rows: list[dict[str, float | int | str]] = []
    for round_index in range(rounds + 1):
        labeled_local = np.asarray(sorted(labeled), dtype=int)
        labeled_global = pool_idx[labeled_local]

        model = make_active_learning_model(random_state + round_index)
        model.fit(x.iloc[labeled_global], y[labeled_global])
        probabilities = model.predict_proba(x.iloc[test_idx])[:, 1]
        metrics = _binary_metrics(y[test_idx], probabilities)
        rows.append(
            {
                "round": round_index,
                "strategy": strategy,
                "labels_used": int(len(labeled_global)),
                "positive_labels": int(y[labeled_global].sum()),
                **metrics,
                "top_fraction_capture": top_fraction_capture(
                    y[test_idx],
                    probabilities,
                    inspection_fraction=inspection_fraction,
                ),
            }
        )

        if round_index == rounds:
            break
        unlabeled = np.asarray(
            sorted(set(range(len(pool_idx))) - labeled),
            dtype=int,
        )
        if len(unlabeled) == 0:
            break
        queried = _query_batch(
            model,
            x_pool,
            unlabeled,
            labeled_local,
            batch_size=query_batch_size,
            strategy=strategy,
            rng=rng,
            candidate_pool_size=candidate_pool_size,
        )
        labeled.update(map(int, queried))

    return pd.DataFrame(rows)


def benchmark_active_learning_strategies(
    frame: pd.DataFrame,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    strategies: Iterable[str] = ("random", "uncertainty", "diversity", "hybrid"),
    **kwargs: object,
) -> pd.DataFrame:
    curves = [
        active_learning_curve(
            frame,
            pool_idx,
            test_idx,
            strategy=strategy,
            **kwargs,
        )
        for strategy in strategies
    ]
    return pd.concat(curves, ignore_index=True)
