from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


@dataclass(frozen=True)
class FeatureScreen:
    eligible: list[str]
    medians: pd.Series
    ranking: pd.DataFrame


def fit_feature_screen(train: pd.DataFrame, feature_names: list[str], *, max_missing_rate: float = 0.50) -> FeatureScreen:
    X = train[feature_names].copy()
    missing = X.isna().mean()
    eligible = missing[missing <= max_missing_rate].index.tolist()
    eligible = [c for c in eligible if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    if not eligible:
        raise ValueError("No process variables survived train-only quality filtering.")
    medians = X[eligible].median()
    filled = X[eligible].fillna(medians)
    scores, _ = f_classif(filled, train["is_fail"].to_numpy())
    scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.finfo(float).max)
    ranking = pd.DataFrame({
        "feature": eligible,
        "anova_f": scores,
        "fit_missing_rate": missing.loc[eligible].to_numpy(),
    }).sort_values(["anova_f", "feature"], ascending=[False, True], ignore_index=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return FeatureScreen(eligible, medians, ranking)


def panel_features(screen: FeatureScreen, panel_size: int | str) -> list[str]:
    if isinstance(panel_size, str):
        if panel_size.lower() != "all":
            raise ValueError(f"Unknown panel size: {panel_size}")
        return screen.ranking["feature"].tolist()
    k = max(1, min(int(panel_size), len(screen.ranking)))
    return screen.ranking.head(k)["feature"].tolist()


def drift_report(history: pd.DataFrame, future: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    rows = []
    for col in selected_features:
        h, f = history[col], future[col]
        h_med, f_med = float(h.median()), float(f.median())
        q1, q3 = float(h.quantile(0.25)), float(h.quantile(0.75))
        iqr = q3 - q1
        robust_shift = (f_med - h_med) / iqr if np.isfinite(iqr) and iqr > 0 else np.nan
        rows.append({
            "feature": col,
            "history_missing_rate": float(h.isna().mean()),
            "test_missing_rate": float(f.isna().mean()),
            "missing_rate_change": float(f.isna().mean() - h.isna().mean()),
            "history_median": h_med,
            "test_median": f_med,
            "history_iqr": iqr,
            "robust_median_shift_iqr": float(robust_shift) if np.isfinite(robust_shift) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.assign(abs_robust_shift=out["robust_median_shift_iqr"].abs()).sort_values(
        ["abs_robust_shift", "feature"], ascending=[False, True], ignore_index=True
    )
