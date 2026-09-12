from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionResult:
    task: str
    model_name: str
    features: list[str]
    oof_prediction: np.ndarray
    metrics: dict[str, float]


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=5.0)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    return {"ridge": ridge, "random_forest": rf, "extra_trees": extra}


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
    }


def grouped_oof(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
) -> np.ndarray:
    unique_groups = pd.Series(groups).nunique()
    n_splits = min(int(n_splits), int(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two unique condition groups are required")
    splitter = GroupKFold(n_splits=n_splits)
    prediction = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(X, y, groups):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction[valid_idx] = fitted.predict(X.iloc[valid_idx])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Grouped OOF prediction did not cover every run")
    return prediction


def evaluate_task(
    frame: pd.DataFrame,
    *,
    task: str,
    features: list[str],
    target: str,
    candidates: list[str],
    random_state: int = 42,
    n_splits: int = 5,
) -> list[RegressionResult]:
    usable = frame.dropna(subset=[target]).reset_index(drop=True)
    X = usable[features].copy()
    y = usable[target].astype(float)
    groups = usable["condition_key"].astype(str)
    models = build_candidates(random_state)
    results: list[RegressionResult] = []
    for name in candidates:
        if name not in models:
            raise KeyError(f"Unknown model candidate: {name}")
        pred = grouped_oof(models[name], X, y, groups, n_splits=n_splits)
        results.append(RegressionResult(task, name, features, pred, regression_metrics(y.to_numpy(), pred)))
    return results


def choose_best(results: list[RegressionResult]) -> RegressionResult:
    if not results:
        raise ValueError("No regression results supplied")
    return min(results, key=lambda r: (r.metrics["mae"], r.metrics["rmse"], r.model_name))


def fit_selected(name: str, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Pipeline:
    model = build_candidates(random_state)[name]
    model.fit(X, y)
    return model


def feature_importance_table(model: Pipeline, feature_names: list[str], task: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).reshape(-1))
    else:
        values = np.full(len(feature_names), np.nan)
    return pd.DataFrame({
        "task": task,
        "feature": feature_names,
        "importance": values,
    }).sort_values(["task", "importance"], ascending=[True, False])
