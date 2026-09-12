from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    panel_size: int
    feature_names: list[str]
    pipeline: Pipeline
    probabilities: np.ndarray
    metrics: dict[str, float]


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=4000, random_state=random_state)),
    ])
    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state)),
    ])
    return {"logistic_regression": logistic, "extra_trees": extra}


def ranking_metrics(y_true, probabilities) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    result = {"average_precision": float(average_precision_score(y, p)), "brier_score": float(brier_score_loss(y, p))}
    result["roc_auc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result


def threshold_metrics(y_true, probabilities, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bal_acc = float(balanced_accuracy_score(y, pred))
    out: dict[str, float | int] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": bal_acc,
        "balanced_error_rate": float(1.0 - bal_acc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    out.update(ranking_metrics(y, p))
    return out


def choose_threshold(y_true, probabilities, min_recall: float = 0.70) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best_threshold, best_precision, best_mcc = 0.5, -1.0, -np.inf
    for threshold in candidates:
        pred = p >= threshold
        recall = recall_score(y, pred, zero_division=0)
        if recall + 1e-12 < min_recall:
            continue
        precision = precision_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        if precision > best_precision or (np.isclose(precision, best_precision) and mcc > best_mcc):
            best_precision, best_mcc, best_threshold = precision, mcc, float(threshold)
    return best_threshold if best_precision >= 0 else 0.0


def evaluate_candidate(model, train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str], model_name: str) -> CandidateResult:
    fitted = clone(model)
    fitted.fit(train[feature_names], train["is_fail"])
    probabilities = fitted.predict_proba(valid[feature_names])[:, 1]
    return CandidateResult(model_name, len(feature_names), list(feature_names), fitted, probabilities, ranking_metrics(valid["is_fail"], probabilities))


def choose_compact_candidate(results: list[CandidateResult], *, average_precision_tolerance: float = 0.02) -> CandidateResult:
    if not results:
        raise ValueError("No candidate results.")
    best_ap = max(r.metrics["average_precision"] for r in results)
    eligible = [r for r in results if r.metrics["average_precision"] >= best_ap - average_precision_tolerance]
    return min(eligible, key=lambda r: (r.panel_size, -r.metrics["average_precision"], r.metrics["brier_score"], r.model_name))
