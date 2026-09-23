from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import ExtraTreesRegressor

FEATURES = ["heating_rate_c_min", "soak_temp_c", "soak_duration_min", "cooling_rate_c_min", "carbon_pct"]
TARGETS = ["hardness_hrc", "tensile_mpa", "distortion_mm", "energy_kwh"]
BOUNDS = {
    "heating_rate_c_min": (2.0, 15.0),
    "soak_temp_c": (780.0, 980.0),
    "soak_duration_min": (10.0, 90.0),
    "cooling_rate_c_min": (5.0, 60.0),
    "carbon_pct": (0.2, 0.8),
}

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_process(X, seed=42, noisy=True):
    rng = np.random.default_rng(seed)
    x = X.copy()
    austenite = _sigmoid((x.soak_temp_c - 825) / 28) * _sigmoid((x.soak_duration_min - 18) / 10)
    temper_penalty = np.maximum(x.soak_temp_c - 930, 0) / 70
    hardness = 18 + 28 * austenite + 20 * x.carbon_pct + 0.17 * x.cooling_rate_c_min - 7 * temper_penalty
    tensile = 510 + 10.5 * hardness - 0.9 * np.maximum(x.cooling_rate_c_min - 48, 0) ** 1.4
    distortion = 0.08 + 0.0055 * x.cooling_rate_c_min + 0.008 * x.heating_rate_c_min + 0.0018 * np.abs(x.soak_temp_c - 870) + 0.001 * x.soak_duration_min
    energy = 2.0 + 0.018 * (x.soak_temp_c - 20) + 0.025 * x.soak_duration_min + 0.05 * x.heating_rate_c_min
    if noisy:
        hardness += rng.normal(0, 0.8, len(x))
        tensile += rng.normal(0, 16, len(x))
        distortion += rng.normal(0, 0.018, len(x))
    return pd.DataFrame({
        "hardness_hrc": np.clip(hardness, 0, 70),
        "tensile_mpa": np.clip(tensile, 0, None),
        "distortion_mm": np.clip(distortion, 0, None),
        "energy_kwh": energy,
    })

def generate_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({k: rng.uniform(*v, n) for k, v in BOUNDS.items()})
    return pd.concat([X, simulate_process(X, seed + 1)], axis=1)

def fit_surrogates(df, seed=42):
    out = {}
    for t in TARGETS:
        m = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        m.fit(df[FEATURES], df[t])
        out[t] = m
    return out

@dataclass(frozen=True)
class Recommendation:
    settings: dict[str, float]
    predicted: dict[str, float]
    objective: float

def recommend_settings(models, carbon_pct=0.45, min_hardness=48.0, min_tensile=950.0, seed=42):
    controls = FEATURES[:-1]
    bounds = [BOUNDS[c] for c in controls]
    def pred(z):
        row = pd.DataFrame([[*z, carbon_pct]], columns=FEATURES)
        return {t: float(models[t].predict(row)[0]) for t in TARGETS}
    def obj(z):
        p = pred(z)
        pen = 80 * max(0, min_hardness - p["hardness_hrc"]) ** 2 + 0.02 * max(0, min_tensile - p["tensile_mpa"]) ** 2
        return 80 * p["distortion_mm"] + 1.8 * p["energy_kwh"] - 0.5 * p["hardness_hrc"] - 0.01 * p["tensile_mpa"] + pen
    r = differential_evolution(obj, bounds, seed=seed, maxiter=6, popsize=4, polish=True)
    return Recommendation(dict(zip(controls, map(float, r.x))), pred(r.x), float(r.fun))

if __name__ == "__main__":
    print(recommend_settings(fit_surrogates(generate_dataset())))
