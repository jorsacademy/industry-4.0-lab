from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
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


@dataclass
class ThresholdResult:
    threshold: float
    method: str
    precision: float
    recall: float


def screen_features(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_columns: list[str],
    max_missing_rate: float,
) -> list[str]:
    fit = frame.loc[rows, candidate_columns]
    missing = fit.isna().mean()
    eligible = missing[missing <= float(max_missing_rate)].index.tolist()
    kept: list[str] = []
    for col in eligible:
        if fit[col].nunique(dropna=True) > 1:
            kept.append(col)
    if not kept:
        raise ValueError("No eligible predictor columns after train-only screening")
    return kept


def rank_features(
    frame: pd.DataFrame,
    target: pd.Series,
    rows: np.ndarray,
    columns: list[str],
    random_state: int,
    n_estimators: int,
    sample_rows: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sample = np.asarray(rows, dtype=int)
    if len(sample) > int(sample_rows):
        positives = sample[target.iloc[sample].to_numpy() == 1]
        negatives = sample[target.iloc[sample].to_numpy() == 0]
        remaining = max(0, int(sample_rows) - len(positives))
        if remaining < len(negatives):
            negatives = rng.choice(negatives, size=remaining, replace=False)
        sample = np.concatenate([positives, negatives])
        rng.shuffle(sample)

    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(frame.loc[sample, columns]).astype(np.float32, copy=False)
    y = target.iloc[sample].to_numpy(dtype=np.int8)
    model = ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        min_samples_leaf=5,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x, y)
    ranking = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
    return ranking.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)


def make_model(
    name: str,
    random_state: int,
    extra_trees: int,
    min_samples_leaf: int,
) -> Pipeline:
    if name == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=400,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=int(extra_trees),
                        min_samples_leaf=int(min_samples_leaf),
                        class_weight="balanced",
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {name}")


def probability(model: Pipeline, frame: pd.DataFrame, rows: np.ndarray, columns: list[str]) -> np.ndarray:
    return model.predict_proba(frame.loc[rows, columns])[:, 1]


def ranking_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    result = {
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        result["roc_auc"] = float("nan")
    return result


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, min_recall: float) -> ThresholdResult:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return ThresholdResult(0.5, "fallback_0.5", 0.0, 0.0)

    feasible = np.flatnonzero(recall[:-1] >= float(min_recall))
    if len(feasible):
        # Maximize precision; break ties with the stricter threshold.
        best = max(feasible, key=lambda i: (precision[i], thresholds[i]))
        return ThresholdResult(
            float(thresholds[best]),
            "max_precision_at_min_recall",
            float(precision[best]),
            float(recall[best]),
        )

    beta2 = 4.0
    denom = beta2 * precision[:-1] + recall[:-1]
    f2 = np.divide(
        (1.0 + beta2) * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best = int(np.nanargmax(f2))
    return ThresholdResult(
        float(thresholds[best]),
        "max_f2_recall_constraint_unreachable",
        float(precision[best]),
        float(recall[best]),
    )


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= float(threshold)).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "prevalence": float(np.mean(y_true)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alarms_per_1000_rows": float(pred.mean() * 1000.0),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics


def drift_diagnostics(
    frame: pd.DataFrame,
    fit_rows: np.ndarray,
    test_rows: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for col in columns:
        a = pd.to_numeric(frame.loc[fit_rows, col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(frame.loc[test_rows, col], errors="coerce").dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        q25, q75 = np.quantile(a, [0.25, 0.75])
        iqr = float(q75 - q25)
        shift = float((np.median(b) - np.median(a)) / iqr) if iqr > 1e-12 else float("nan")
        records.append(
            {
                "feature": col,
                "ks_statistic": float(ks_2samp(a, b, method="auto").statistic),
                "fit_missing_rate": float(1.0 - len(a) / len(fit_rows)),
                "test_missing_rate": float(1.0 - len(b) / len(test_rows)),
                "fit_median": float(np.median(a)),
                "test_median": float(np.median(b)),
                "fit_iqr": iqr,
                "robust_median_shift": shift,
            }
        )
    return pd.DataFrame(records).sort_values("ks_statistic", ascending=False).reset_index(drop=True)


def fitted_feature_importance(model: Pipeline, columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        values = np.full(len(columns), np.nan)
    out = pd.DataFrame({"feature": columns, "importance": values})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
