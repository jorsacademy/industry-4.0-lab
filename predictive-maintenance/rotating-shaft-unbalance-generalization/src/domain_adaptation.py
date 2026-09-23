from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CoralTransform:
    source_mean: np.ndarray
    target_mean: np.ndarray
    transform: np.ndarray
    medians: np.ndarray

    def transform_source(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, self.medians)
        return (x - self.source_mean) @ self.transform + self.target_mean

    def transform_target(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), x, self.medians)


def _matrix_sqrt_psd(matrix: np.ndarray, *, inverse: bool, eps: float) -> np.ndarray:
    matrix = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eps, None)
    exponent = -0.5 if inverse else 0.5
    return (vectors * (values**exponent)) @ vectors.T


def fit_coral_transform(
    source_x: np.ndarray,
    target_x: np.ndarray,
    *,
    regularization: float = 1e-5,
) -> CoralTransform:
    source_x = np.asarray(source_x, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    if source_x.ndim != 2 or target_x.ndim != 2 or source_x.shape[1] != target_x.shape[1]:
        raise ValueError("source_x and target_x must be 2D with equal feature counts")
    if len(source_x) < 2 or len(target_x) < 2:
        raise ValueError("CORAL needs at least two source and target rows")
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    stacked = np.vstack([source_x, target_x])
    medians = np.nanmedian(stacked, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    source = np.where(np.isfinite(source_x), source_x, medians)
    target = np.where(np.isfinite(target_x), target_x, medians)

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_cov = np.cov(source - source_mean, rowvar=False) + regularization * np.eye(source.shape[1])
    target_cov = np.cov(target - target_mean, rowvar=False) + regularization * np.eye(target.shape[1])

    transform = _matrix_sqrt_psd(
        source_cov,
        inverse=True,
        eps=regularization,
    ) @ _matrix_sqrt_psd(
        target_cov,
        inverse=False,
        eps=regularization,
    )
    return CoralTransform(source_mean, target_mean, transform, medians)


def chronological_target_split(
    target: pd.DataFrame,
    *,
    adaptation_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < adaptation_fraction < 1:
        raise ValueError("adaptation_fraction must be in (0, 1)")
    if "recording" not in target or "window_index" not in target:
        raise ValueError("target frame must contain recording and window_index")

    adaptation: list[int] = []
    evaluation: list[int] = []
    for _, group in target.groupby("recording", sort=True):
        ordered = group.sort_values("window_index")
        cut = max(1, int(np.floor(len(ordered) * adaptation_fraction)))
        cut = min(cut, len(ordered) - 1)
        adaptation.extend(map(int, ordered.index[:cut]))
        evaluation.extend(map(int, ordered.index[cut:]))
    return np.asarray(sorted(adaptation), dtype=int), np.asarray(sorted(evaluation), dtype=int)


def _make_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "severity_mae": float(np.mean(np.abs(prediction - y_true))),
    }


def _sample_k_per_class(
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    selected: list[int] = []
    for label in sorted(np.unique(y)):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < k:
            raise ValueError(f"class {label} has fewer than {k} target-adaptation rows")
        selected.extend(map(int, rng.choice(candidates, size=k, replace=False)))
    return np.asarray(sorted(selected), dtype=int)


def domain_adaptation_benchmark(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    source_session: str = "D",
    target_session: str = "E",
    adaptation_fraction: float = 0.25,
    k_values: Iterable[int] = (1, 3, 5, 10),
    repeats: int = 5,
    target_weight: float = 8.0,
    random_state: int = 42,
) -> pd.DataFrame:
    source = frame[frame["session"] == source_session].reset_index(drop=True)
    target = frame[frame["session"] == target_session].reset_index(drop=True)
    if source.empty or target.empty:
        raise ValueError("source and target sessions must both be present")

    adaptation_idx, evaluation_idx = chronological_target_split(
        target,
        adaptation_fraction=adaptation_fraction,
    )

    source_x = source[feature_columns].to_numpy(dtype=float)
    source_y = source["severity"].to_numpy(dtype=int)
    target_x = target[feature_columns].to_numpy(dtype=float)
    target_y = target["severity"].to_numpy(dtype=int)

    classes = set(np.unique(source_y))
    if classes != set(np.unique(target_y[adaptation_idx])) or classes != set(np.unique(target_y[evaluation_idx])):
        raise ValueError("source, target-adaptation, and target-evaluation partitions must contain the same classes")

    coral = fit_coral_transform(source_x, target_x[adaptation_idx])
    aligned_source = coral.transform_source(source_x)
    target_space = coral.transform_target(target_x)

    rows: list[dict[str, float | int | str]] = []

    source_only = _make_model(random_state)
    source_only.fit(source_x, source_y)
    prediction = source_only.predict(target_x[evaluation_idx])
    rows.append(
        {
            "method": "source_only",
            "k_per_class": 0,
            "repeat": 0,
            "target_labels": 0,
            **_metrics(target_y[evaluation_idx], prediction),
        }
    )

    coral_model = _make_model(random_state + 1)
    coral_model.fit(aligned_source, source_y)
    prediction = coral_model.predict(target_space[evaluation_idx])
    rows.append(
        {
            "method": "coral_unsupervised",
            "k_per_class": 0,
            "repeat": 0,
            "target_labels": 0,
            **_metrics(target_y[evaluation_idx], prediction),
        }
    )

    rng = np.random.default_rng(random_state)
    adaptation_y = target_y[adaptation_idx]
    for k in map(int, k_values):
        for repeat in range(repeats):
            local = _sample_k_per_class(adaptation_y, k, rng)
            labeled_target_idx = adaptation_idx[local]

            target_only = _make_model(random_state + 1000 + 100 * k + repeat)
            target_only.fit(target_space[labeled_target_idx], target_y[labeled_target_idx])
            prediction = target_only.predict(target_space[evaluation_idx])
            rows.append(
                {
                    "method": "target_only_few_shot",
                    "k_per_class": k,
                    "repeat": repeat,
                    "target_labels": int(len(labeled_target_idx)),
                    **_metrics(target_y[evaluation_idx], prediction),
                }
            )

            combined_x = np.vstack([aligned_source, target_space[labeled_target_idx]])
            combined_y = np.concatenate([source_y, target_y[labeled_target_idx]])
            sample_weight = np.concatenate(
                [
                    np.ones(len(source_y), dtype=float),
                    np.full(len(labeled_target_idx), target_weight, dtype=float),
                ]
            )
            hybrid = _make_model(random_state + 2000 + 100 * k + repeat)
            hybrid.fit(
                combined_x,
                combined_y,
                model__sample_weight=sample_weight,
            )
            prediction = hybrid.predict(target_space[evaluation_idx])
            rows.append(
                {
                    "method": "coral_plus_few_shot",
                    "k_per_class": k,
                    "repeat": repeat,
                    "target_labels": int(len(labeled_target_idx)),
                    **_metrics(target_y[evaluation_idx], prediction),
                }
            )

    return pd.DataFrame(rows)
