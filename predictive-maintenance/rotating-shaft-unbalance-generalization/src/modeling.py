from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import compact_feature_columns, order_feature_columns

LABELS = np.arange(5, dtype=int)


@dataclass(frozen=True)
class Candidate:
    name: str
    representation: str
    estimator: BaseEstimator


def candidates(random_state: int = 42, n_jobs: int = -1) -> list[Candidate]:
    return [
        Candidate(
            "compact_logistic",
            "compact",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]),
        ),
        Candidate(
            "compact_random_forest",
            "compact",
            Pipeline([
                ("model", RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=n_jobs,
                )),
            ]),
        ),
        Candidate(
            "compact_extra_trees",
            "compact",
            Pipeline([
                ("model", ExtraTreesClassifier(
                    n_estimators=350,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=n_jobs,
                )),
            ]),
        ),
        Candidate(
            "order_logistic",
            "order",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=random_state)),
            ]),
        ),
        Candidate(
            "order_mlp",
            "order",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=random_state,
                )),
            ]),
        ),
    ]


def feature_columns(frame: pd.DataFrame, representation: str, sensors: tuple[int, ...] = (1, 2, 3)) -> list[str]:
    if representation == "compact":
        return compact_feature_columns(frame, sensors=sensors)
    if representation == "order":
        return order_feature_columns(frame, sensors=sensors)
    raise ValueError(f"Unknown representation {representation}")


def blocked_splits(frame: pd.DataFrame, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    blocks = frame["dev_block"].to_numpy(dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        test_idx = np.flatnonzero(blocks == fold)
        train_idx = np.flatnonzero(blocks != fold)
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"Fold {fold} is empty")
        splits.append((train_idx, test_idx))
    return splits


def _fold_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(accuracy_score(y_true, y_pred)),
    )


def cross_validate_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    n_folds: int,
    sensors: tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    cols = feature_columns(frame, candidate.representation, sensors=sensors)
    X = frame[cols].to_numpy(dtype=float)
    y = frame["severity"].to_numpy(dtype=int)
    fold_rows: list[dict[str, float | int]] = []
    for fold, (train_idx, test_idx) in enumerate(blocked_splits(frame, n_folds=n_folds)):
        estimator = clone(candidate.estimator)
        estimator.fit(X[train_idx], y[train_idx])
        pred = estimator.predict(X[test_idx])
        macro_f1, balanced, accuracy = _fold_score(y[test_idx], pred)
        fold_rows.append({"fold": fold, "macro_f1": macro_f1, "balanced_accuracy": balanced, "accuracy": accuracy})
    scores = pd.DataFrame(fold_rows)
    return {
        "model": candidate.name,
        "representation": candidate.representation,
        "sensors": ",".join(map(str, sensors)),
        "n_features": len(cols),
        "macro_f1": float(scores["macro_f1"].mean()),
        "macro_f1_std": float(scores["macro_f1"].std(ddof=1)),
        "balanced_accuracy": float(scores["balanced_accuracy"].mean()),
        "accuracy": float(scores["accuracy"].mean()),
    }


def select_candidate(dev: pd.DataFrame, n_folds: int, random_state: int, n_jobs: int) -> tuple[Candidate, pd.DataFrame]:
    candidate_list = candidates(random_state=random_state, n_jobs=n_jobs)
    rows = [cross_validate_candidate(dev, c, n_folds=n_folds) for c in candidate_list]
    table = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(table.iloc[0]["model"])
    selected = next(c for c in candidate_list if c.name == selected_name)
    return selected, table


def fit_calibrated(
    dev: pd.DataFrame,
    candidate: Candidate,
    n_folds: int,
    sensors: tuple[int, ...] = (1, 2, 3),
) -> tuple[CalibratedClassifierCV, list[str]]:
    cols = feature_columns(dev, candidate.representation, sensors=sensors)
    X = dev[cols].to_numpy(dtype=float)
    y = dev["severity"].to_numpy(dtype=int)
    splits = blocked_splits(dev, n_folds=n_folds)
    calibrated = CalibratedClassifierCV(estimator=clone(candidate.estimator), method="sigmoid", cv=splits)
    calibrated.fit(X, y)
    return calibrated, cols


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def evaluate_classifier(
    model: BaseEstimator,
    frame: pd.DataFrame,
    cols: list[str],
    calibration_bins: int = 10,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = frame[cols].to_numpy(dtype=float)
    y = frame["severity"].to_numpy(dtype=int)
    probs = model.predict_proba(X)
    pred = np.asarray(model.classes_)[probs.argmax(axis=1)].astype(int)
    onehot = np.eye(len(LABELS))[y]
    metrics = {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "severity_mae": float(np.mean(np.abs(pred - y))),
        "multiclass_brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "log_loss": float(log_loss(y, probs, labels=LABELS)),
        "ece": expected_calibration_error(y, probs, n_bins=calibration_bins),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=LABELS, zero_division=0
    )
    class_report = pd.DataFrame({
        "severity": LABELS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    matrix = pd.DataFrame(
        confusion_matrix(y, pred, labels=LABELS),
        index=[f"true_{i}" for i in LABELS],
        columns=[f"pred_{i}" for i in LABELS],
    )
    predictions = frame[["recording", "session", "severity", "window_index", "start_sec", "compact__rpm_median"]].copy()
    predictions["prediction"] = pred
    predictions["confidence"] = probs.max(axis=1)
    for idx, klass in enumerate(model.classes_):
        predictions[f"p_{int(klass)}"] = probs[:, idx]
    return metrics, predictions, matrix, class_report


def fit_uncalibrated(dev: pd.DataFrame, candidate: Candidate, cols: list[str]) -> BaseEstimator:
    model = clone(candidate.estimator)
    model.fit(dev[cols].to_numpy(dtype=float), dev["severity"].to_numpy(dtype=int))
    return model


def feature_importance_table(model: BaseEstimator, cols: list[str]) -> pd.DataFrame:
    inner = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if hasattr(inner, "feature_importances_"):
        values = np.asarray(inner.feature_importances_, dtype=float)
    elif hasattr(inner, "coef_"):
        values = np.mean(np.abs(np.asarray(inner.coef_, dtype=float)), axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": cols, "importance": values}).sort_values("importance", ascending=False)


def sensor_ablation_external(
    dev: pd.DataFrame,
    evaluation: pd.DataFrame,
    candidate: Candidate,
    n_folds: int,
    calibration_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensors in ((1,), (2,), (3,), (1, 2, 3)):
        model, cols = fit_calibrated(dev, candidate, n_folds=n_folds, sensors=sensors)
        metrics, _, _, _ = evaluate_classifier(model, evaluation, cols, calibration_bins=calibration_bins)
        rows.append({
            "sensors": "+".join(map(str, sensors)),
            "n_features": len(cols),
            **metrics,
        })
    return pd.DataFrame(rows)


def rpm_band_table(predictions: pd.DataFrame, bands: Iterable[float]) -> pd.DataFrame:
    bands = list(map(float, bands))
    labels = [f"{int(bands[i])}-{int(bands[i+1])}" for i in range(len(bands) - 1)]
    working = predictions.copy()
    working["rpm_band"] = pd.cut(
        working["compact__rpm_median"], bins=bands, labels=labels, right=False, include_lowest=True
    )
    rows: list[dict[str, object]] = []
    for band, group in working.groupby("rpm_band", observed=True):
        y = group["severity"].to_numpy(dtype=int)
        pred = group["prediction"].to_numpy(dtype=int)
        rows.append({
            "rpm_band": str(band),
            "windows": len(group),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "severity_mae": float(np.mean(np.abs(pred - y))),
        })
    return pd.DataFrame(rows)
