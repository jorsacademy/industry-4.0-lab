from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressionMetrics:
    nmae: float
    mean_mae: float
    mean_rmse: float
    mean_r2: float


def target_scale(y_fit: np.ndarray) -> np.ndarray:
    q75 = np.nanpercentile(y_fit, 75, axis=0)
    q25 = np.nanpercentile(y_fit, 25, axis=0)
    iqr = q75 - q25
    std = np.nanstd(y_fit, axis=0)
    fallback = np.where(std > 1e-9, std, 1.0)
    return np.where(iqr > 1e-9, iqr, fallback)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> RegressionMetrics:
    per_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    per_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    per_r2 = np.array([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])])
    return RegressionMetrics(
        nmae=float(np.mean(per_mae / scale)),
        mean_mae=float(np.mean(per_mae)),
        mean_rmse=float(np.mean(per_rmse)),
        mean_r2=float(np.nanmean(per_r2)),
    )


def make_model(name: str, cfg: dict, seed: int):
    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(cfg["ridge_alpha"]))),
        ])
    if name == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(cfg["extra_trees_estimators"]),
                    min_samples_leaf=int(cfg["extra_trees_min_samples_leaf"]),
                    max_features=float(cfg["extra_trees_max_features"]),
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ])
    raise KeyError(name)


def conformal_radius(y_cal: np.ndarray, pred_cal: np.ndarray, coverage: float) -> np.ndarray:
    residuals = np.abs(y_cal - pred_cal)
    n = len(residuals)
    alpha = 1.0 - coverage
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(residuals, q_level, axis=0, method="higher")


def interval_diagnostics(
    y_true: np.ndarray,
    pred: np.ndarray,
    radius: np.ndarray,
    scale: np.ndarray,
) -> pd.DataFrame:
    lower = pred - radius
    upper = pred + radius
    coverage = ((y_true >= lower) & (y_true <= upper)).mean(axis=0)
    return pd.DataFrame({
        "target_index": np.arange(y_true.shape[1]),
        "coverage": coverage,
        "half_width": radius,
        "normalized_half_width": radius / scale,
    })


def block_bootstrap_improvement_ci(
    model_error: np.ndarray,
    baseline_error: np.ndarray,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if model_error.shape != baseline_error.shape:
        raise ValueError("Error arrays must have the same shape")
    diff = baseline_error - model_error
    n = len(diff)
    if n == 0:
        raise ValueError("No observations for bootstrap")
    block_length = max(1, min(block_length, n))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, max(1, n - block_length + 1))
    means = np.empty(repetitions, dtype=float)
    needed = int(np.ceil(n / block_length))
    for i in range(repetitions):
        chosen = rng.choice(starts, size=needed, replace=True)
        sample = np.concatenate([diff[s : s + block_length] for s in chosen])[:n]
        means[i] = float(sample.mean())
    return {
        "mean_improvement": float(diff.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def drift_table(X_fit: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in X_fit.columns:
        a = pd.to_numeric(X_fit[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        b = pd.to_numeric(X_test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        q25, q75 = np.quantile(a, [0.25, 0.75])
        denom = max(float(q75 - q25), float(np.std(a)), 1e-9)
        rows.append({
            "feature": col,
            "ks_statistic": float(ks_2samp(a, b, method="asymp").statistic),
            "robust_median_shift": float(abs(np.median(b) - np.median(a)) / denom),
        })
    if not rows:
        return pd.DataFrame(columns=["feature", "ks_statistic", "robust_median_shift"])
    return pd.DataFrame(rows).sort_values(["ks_statistic", "robust_median_shift"], ascending=False).reset_index(drop=True)


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence: np.ndarray,
    setpoint: np.ndarray,
    scale: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for j in range(y_true.shape[1]):
        rows.append({
            "target_index": j,
            "mae": float(mean_absolute_error(y_true[:, j], y_pred[:, j])),
            "rmse": float(mean_squared_error(y_true[:, j], y_pred[:, j]) ** 0.5),
            "r2": float(r2_score(y_true[:, j], y_pred[:, j])),
            "normalized_mae": float(mean_absolute_error(y_true[:, j], y_pred[:, j]) / scale[j]),
            "persistence_mae": float(mean_absolute_error(y_true[:, j], persistence[:, j])),
            "setpoint_mae": float(mean_absolute_error(y_true[:, j], setpoint[:, j])),
        })
    return pd.DataFrame(rows)


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    transformed_names = list(imputer.get_feature_names_out(feature_names))
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": transformed_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
