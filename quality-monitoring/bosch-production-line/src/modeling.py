from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {"Id", "Response", "start_time", "end_time"}


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    selection: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._logit(probabilities), y)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(probabilities))[:, 1]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def temporal_four_way_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    selection_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> TemporalSplit:
    total = train_fraction + selection_fraction + calibration_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("Split fractions must sum to 1")
    if "start_time" not in frame:
        raise ValueError("Feature table must contain start_time")

    times = pd.to_numeric(frame["start_time"], errors="coerce")
    if times.notna().mean() < 0.95:
        raise ValueError("At least 95% of rows must have a process start time for chronological evaluation")

    finite = times.dropna().to_numpy(dtype=float)
    q1 = np.quantile(finite, train_fraction, method="nearest")
    q2 = np.quantile(finite, train_fraction + selection_fraction, method="nearest")
    q3 = np.quantile(
        finite,
        train_fraction + selection_fraction + calibration_fraction,
        method="nearest",
    )

    train = np.flatnonzero((times <= q1).to_numpy())
    selection = np.flatnonzero(((times > q1) & (times <= q2)).to_numpy())
    calibration = np.flatnonzero(((times > q2) & (times <= q3)).to_numpy())
    test = np.flatnonzero((times > q3).to_numpy())

    if min(map(len, (train, selection, calibration, test))) == 0:
        raise ValueError("Chronological split produced an empty partition")
    return TemporalSplit(train, selection, calibration, test)


def make_model(name: str, random_state: int = 42) -> Pipeline:
    if name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1500,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
    if name == "hist_gradient":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    result = {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = float("nan")
    return result


def select_operating_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    strategy: str = "min_precision",
    min_precision: float = 0.10,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    if strategy == "min_precision":
        eligible = np.flatnonzero(precision[:-1] >= min_precision)
        if eligible.size:
            best_recall = recall[eligible].max()
            candidates = eligible[np.isclose(recall[eligible], best_recall)]
            best = candidates[np.argmax(precision[candidates])]
            return float(thresholds[best])
        return 0.5

    if strategy == "mcc":
        unique = np.unique(np.quantile(probabilities, np.linspace(0.01, 0.99, 199)))
        scores = [matthews_corrcoef(y_true, probabilities >= threshold) for threshold in unique]
        return float(unique[int(np.argmax(scores))])

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        **probability_metrics(y_true, probabilities),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_and_select_model(
    frame: pd.DataFrame,
    split: TemporalSplit,
    candidates: list[str],
    random_state: int,
) -> tuple[str, Pipeline, pd.DataFrame]:
    columns = feature_columns(frame)
    x = frame[columns]
    y = frame["Response"].astype(int).to_numpy()

    comparison: list[dict[str, float | str]] = []
    best_name = ""
    best_model: Pipeline | None = None
    best_ap = -np.inf
    for name in candidates:
        model = make_model(name, random_state=random_state)
        model.fit(x.iloc[split.train], y[split.train])
        probabilities = model.predict_proba(x.iloc[split.selection])[:, 1]
        metrics = probability_metrics(y[split.selection], probabilities)
        comparison.append({"model": name, **metrics})
        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model candidate was fitted")
    return best_name, best_model, pd.DataFrame(comparison).sort_values("average_precision", ascending=False)
