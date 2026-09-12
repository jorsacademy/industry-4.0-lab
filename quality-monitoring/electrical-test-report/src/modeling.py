from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CVResult:
    model_name: str
    probabilities: np.ndarray
    metrics: dict[str, float]


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)),
    ])
    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state)),
    ])
    return {"logistic_regression": linear, "random_forest": rf, "extra_trees": extra}


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    result = {
        "average_precision": float(average_precision_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return result


def out_of_fold_probabilities(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5, random_state: int = 42):
    unique_groups = pd.Series(groups).nunique()
    n_splits = max(2, min(int(n_splits), int(unique_groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prob = np.full(len(y), np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups)):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_id[valid_idx] = fold
    if np.isnan(prob).any():
        raise RuntimeError("OOF prediction did not cover all records.")
    return prob, fold_id


def evaluate_candidates(X, y, groups, *, n_splits: int = 5, random_state: int = 42) -> list[CVResult]:
    results: list[CVResult] = []
    for name, model in build_candidates(random_state).items():
        prob, _ = out_of_fold_probabilities(model, X, y, groups, n_splits=n_splits, random_state=random_state)
        results.append(CVResult(name, prob, _metrics(np.asarray(y), prob)))
    return results


def choose_threshold_for_recall(y_true, probabilities, target_recall: float = 0.90) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(p)
    best = 0.5
    best_precision = -1.0
    for t in candidates:
        pred = p >= t
        recall = recall_score(y, pred, zero_division=0)
        if recall >= target_recall:
            precision = precision_score(y, pred, zero_division=0)
            if precision > best_precision or (precision == best_precision and t > best):
                best_precision = precision
                best = float(t)
    return best


def feature_importance_table(fitted_pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False)
