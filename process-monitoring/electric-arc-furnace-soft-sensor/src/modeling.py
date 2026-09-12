from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def chronological_blocks(frame: pd.DataFrame, fractions: tuple[float, float, float, float]) -> dict[str, pd.DataFrame]:
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Chronological split fractions must sum to 1")
    ordered = frame.sort_values(["target_time", "HEATID"]).reset_index(drop=True)
    n = len(ordered)
    if n < 40:
        raise ValueError(f"Need at least 40 heat snapshots, got {n}")
    cut1 = max(1, int(n * fractions[0]))
    cut2 = max(cut1 + 1, int(n * (fractions[0] + fractions[1])))
    cut3 = max(cut2 + 1, int(n * (fractions[0] + fractions[1] + fractions[2])))
    cut3 = min(cut3, n - 1)
    return {
        "fit": ordered.iloc[:cut1].copy(),
        "selection": ordered.iloc[cut1:cut2].copy(),
        "calibration": ordered.iloc[cut2:cut3].copy(),
        "test": ordered.iloc[cut3:].copy(),
    }


def candidate_models(seed: int, n_estimators: int, min_samples_leaf: int, max_features: float, ridge_alpha: float):
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge_alpha)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(residual)),
        "within_10c": float(np.mean(np.abs(residual) <= 10.0)),
        "within_20c": float(np.mean(np.abs(residual) <= 20.0)),
    }


def conformal_radius(y_true, y_pred, coverage: float) -> float:
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    if len(residuals) == 0:
        raise ValueError("empty calibration set")
    quantile = min(1.0, math.ceil((len(residuals) + 1) * coverage) / len(residuals))
    return float(np.quantile(residuals, quantile, method="higher"))


def interval_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def model_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    imputer = model.named_steps["imputer"]
    try:
        names = list(imputer.get_feature_names_out(feature_names))
    except Exception:
        names = feature_names
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    n = min(len(names), len(values))
    return pd.DataFrame({"feature": names[:n], "importance": values[:n]}).sort_values("importance", ascending=False)
