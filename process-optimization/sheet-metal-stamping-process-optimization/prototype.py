from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import ExtraTreesRegressor

FEATURES = ["press_force_kn", "ram_speed_mm_s", "blank_holder_force_kn", "lubrication", "thickness_mm", "yield_strength_mpa"]
TARGETS = ["springback_deg", "thinning_pct", "wrinkle_risk", "crack_risk", "energy_index"]
BOUNDS = {
    "press_force_kn": (300.0, 900.0),
    "ram_speed_mm_s": (40.0, 160.0),
    "blank_holder_force_kn": (40.0, 180.0),
    "lubrication": (0.1, 1.0),
    "thickness_mm": (0.8, 2.0),
    "yield_strength_mpa": (220.0, 500.0),
}

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def simulate_process(X: pd.DataFrame, seed: int = 42, noisy: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = X.copy()
    force_ratio = x.press_force_kn / (x.yield_strength_mpa * x.thickness_mm + 1e-6)
    holder_ratio = x.blank_holder_force_kn / (x.press_force_kn + 1e-6)
    springback = 7.5 + 0.018 * x.yield_strength_mpa - 4.2 * force_ratio - 1.8 * x.lubrication + 0.012 * x.ram_speed_mm_s
    thinning = 8.0 + 7.0 * force_ratio + 0.035 * x.ram_speed_mm_s - 8.0 * x.lubrication + 12.0 * np.maximum(holder_ratio - 0.16, 0)
    wrinkle = _sigmoid(2.8 - 22 * holder_ratio - 1.4 * x.lubrication - 0.004 * x.press_force_kn)
    crack = _sigmoid(-5.2 + 0.32 * thinning + 0.005 * (x.yield_strength_mpa - 300) + 0.006 * (x.ram_speed_mm_s - 80))
    energy = (x.press_force_kn / 600.0) ** 1.25 * (0.55 + x.ram_speed_mm_s / 240.0)
    if noisy:
        springback += rng.normal(0, 0.25, len(x))
        thinning += rng.normal(0, 0.6, len(x))
        wrinkle = np.clip(wrinkle + rng.normal(0, 0.015, len(x)), 0, 1)
        crack = np.clip(crack + rng.normal(0, 0.015, len(x)), 0, 1)
    return pd.DataFrame({
        "springback_deg": np.clip(springback, 0, None),
        "thinning_pct": np.clip(thinning, 0, None),
        "wrinkle_risk": wrinkle,
        "crack_risk": crack,
        "energy_index": energy,
    })

def generate_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({k: rng.uniform(*v, n) for k, v in BOUNDS.items()})
    return pd.concat([X, simulate_process(X, seed + 1, True)], axis=1)

def fit_surrogates(df, seed=42):
    models = {}
    for t in TARGETS:
        m = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        m.fit(df[FEATURES], df[t])
        models[t] = m
    return models

@dataclass(frozen=True)
class Recommendation:
    settings: dict[str, float]
    predicted: dict[str, float]
    objective: float

def recommend_settings(models, thickness_mm=1.2, yield_strength_mpa=350.0, seed=42):
    controls = FEATURES[:4]
    bounds = [BOUNDS[c] for c in controls]
    def pred(z):
        row = pd.DataFrame([[*z, thickness_mm, yield_strength_mpa]], columns=FEATURES)
        return {t: float(models[t].predict(row)[0]) for t in TARGETS}
    def obj(z):
        p = pred(z)
        penalty = 80 * max(0, p["thinning_pct"] - 20) ** 2
        penalty += 900 * max(0, p["crack_risk"] - 0.25) ** 2 + 900 * max(0, p["wrinkle_risk"] - 0.25) ** 2
        return 3 * p["springback_deg"] + 1.5 * p["thinning_pct"] + 120 * (p["crack_risk"] + p["wrinkle_risk"]) + 8 * p["energy_index"] + penalty
    r = differential_evolution(obj, bounds, seed=seed, maxiter=6, popsize=4, polish=True)
    p = pred(r.x)
    return Recommendation(dict(zip(controls, map(float, r.x))), p, float(r.fun))

if __name__ == "__main__":
    print(recommend_settings(fit_surrogates(generate_dataset())))
