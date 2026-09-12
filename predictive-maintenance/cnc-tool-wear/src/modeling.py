from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    metrics: dict[str, object]
    window_predictions: pd.DataFrame
    run_predictions: pd.DataFrame


def encode_binary_target(values: pd.Series, positive_label: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    positive = positive_label.strip().lower()
    unique = sorted(normalized.dropna().unique().tolist())
    if positive not in unique:
        raise ValueError(f"Positive label {positive_label!r} not found. Available labels: {unique}")
    if len(unique) != 2:
        raise ValueError(f"Binary target expected, found labels: {unique}")
    return normalized.eq(positive).astype(int)


def make_splitter(y: pd.Series, groups: pd.Series, maximum_splits: int, random_state: int):
    run_labels = (
        pd.DataFrame({"y": y.to_numpy(), "group": groups.to_numpy()})
        .drop_duplicates("group")
        .groupby("y")["group"]
        .nunique()
    )
    if len(run_labels) != 2 or run_labels.min() < 2:
        raise ValueError("At least two independent experiment groups per class are required.")
    n_splits = int(min(maximum_splits, run_labels.min()))
    n_splits = max(2, n_splits)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def feature_columns(feature_table: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {"experiment_id", "window_start", "window_end", "target"}
    candidates = [c for c in feature_table.columns if c not in excluded]
    categorical = [c for c in candidates if not pd.api.types.is_numeric_dtype(feature_table[c])]
    numeric = [c for c in candidates if c not in categorical]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str], *, scale_numeric: bool):
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = [("numeric", numeric_pipeline, numeric)]
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def candidate_models(numeric: list[str], categorical: list[str], random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=0.5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical, scale_numeric=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def binary_metrics(y_true: Iterable[int], probability: Iterable[float], threshold: float) -> dict[str, object]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    probability_array = np.asarray(list(probability), dtype=float)
    y_pred = (probability_array >= threshold).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    if len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, probability_array))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    feature_table: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    maximum_splits: int,
    random_state: int,
    threshold: float,
) -> EvaluationResult:
    numeric, categorical = feature_columns(feature_table)
    X = feature_table[numeric + categorical]
    splitter = make_splitter(y, groups, maximum_splits, random_state)

    fold_predictions: list[pd.DataFrame] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        probability = model.predict_proba(X.iloc[test_index])[:, 1]
        fold_frame = feature_table.iloc[test_index][
            ["experiment_id", "window_start", "window_end"]
        ].copy()
        fold_frame["fold"] = fold
        fold_frame["y_true"] = y.iloc[test_index].to_numpy()
        fold_frame["probability"] = probability
        fold_predictions.append(fold_frame)

    window_predictions = pd.concat(fold_predictions, ignore_index=True)
    run_predictions = (
        window_predictions.groupby("experiment_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            probability=("probability", "mean"),
            median_probability=("probability", "median"),
            window_count=("probability", "size"),
            fold=("fold", "first"),
        )
        .sort_values("experiment_id")
        .reset_index(drop=True)
    )

    metrics = {
        "run_level": binary_metrics(run_predictions["y_true"], run_predictions["probability"], threshold),
        "window_level": binary_metrics(window_predictions["y_true"], window_predictions["probability"], threshold),
        "independent_runs": int(run_predictions["experiment_id"].nunique()),
        "windows": int(len(window_predictions)),
        "cv_splits": int(window_predictions["fold"].nunique()),
    }
    return EvaluationResult(name, metrics, window_predictions, run_predictions)


def choose_best(results: list[EvaluationResult]) -> EvaluationResult:
    if not results:
        raise ValueError("No model evaluation results supplied.")

    def score(result: EvaluationResult) -> tuple[float, float, float]:
        run = result.metrics["run_level"]
        roc_auc = run.get("roc_auc")
        return (
            float(roc_auc) if roc_auc is not None else -1.0,
            float(run["balanced_accuracy"]),
            float(run["f1"]),
        )

    return max(results, key=score)


def extract_feature_importance(fitted_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
