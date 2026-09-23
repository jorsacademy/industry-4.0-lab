from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["laser_power_w", "cutting_speed_mm_min", "gas_pressure_bar", "focus_mm", "thickness_mm"]
TARGETS = ["roughness_um", "burr_mm", "kerf_mm", "energy_kwh_m"]
BOUNDS = {
    "laser_power_w": (800.0, 3000.0),
    "cutting_speed_mm_min": (500.0, 3000.0),
    "gas_pressure_bar": (4.0, 16.0),
    "focus_mm": (-2.0, 2.0),
    "thickness_mm": (1.0, 8.0),
}

def simulate_process(X, seed=42, noisy=True):
    rng = np.random.default_rng(seed)
    x = X.copy()
    speed_m_s = x.cutting_speed_mm_min / 60000.0
    specific = x.laser_power_w / (x.cutting_speed_mm_min * x.thickness_mm)
    optimum = 0.23 + 0.018 * x.thickness_mm
    mismatch = (specific - optimum) / (0.10 + 0.01 * x.thickness_mm)
    rough = 1.2 + 2.8 * mismatch ** 2 + 0.22 * (x.gas_pressure_bar - 10) ** 2 / 9 + 0.8 * x.focus_mm ** 2
    burr = 0.05 + 0.34 * np.maximum(-mismatch, 0) ** 1.5 + 0.018 * np.maximum(8 - x.gas_pressure_bar, 0) + 0.05 * np.abs(x.focus_mm)
    kerf = 0.11 + 0.00007 * x.laser_power_w - 0.000025 * x.cutting_speed_mm_min + 0.025 * np.abs(x.focus_mm) + 0.008 * x.thickness_mm
    energy = x.laser_power_w / 1000.0 / (speed_m_s * 3600 + 1e-9) / 1000
    if noisy:
        rough += rng.normal(0, 0.12, len(x))
        burr += rng.normal(0, 0.008, len(x))
        kerf += rng.normal(0, 0.006, len(x))
    return pd.DataFrame({
        "roughness_um": np.clip(rough, 0.2, None),
        "burr_mm": np.clip(burr, 0, None),
        "kerf_mm": np.clip(kerf, 0.05, None),
        "energy_kwh_m": energy,
    })

def generate_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({k: rng.uniform(*v, n) for k, v in BOUNDS.items()})
    return pd.concat([X, simulate_process(X, seed + 1)], axis=1)

def fit_surrogates(df, seed=42):
    out = {}
    for t in TARGETS:
        m = RandomForestRegressor(n_estimators=40, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        m.fit(df[FEATURES], df[t])
        out[t] = m
    return out

@dataclass(frozen=True)
class Recommendation:
    settings: dict[str, float]
    predicted: dict[str, float]
    objective: float

def recommend_settings(models, thickness_mm=4.0, seed=42):
    controls = FEATURES[:-1]
    bounds = [BOUNDS[c] for c in controls]
    def pred(z):
        row = pd.DataFrame([[*z, thickness_mm]], columns=FEATURES)
        return {t: float(models[t].predict(row)[0]) for t in TARGETS}
    def obj(z):
        p = pred(z)
        penalty = 600 * max(0, p["burr_mm"] - 0.16) ** 2 + 120 * max(0, p["roughness_um"] - 3.5) ** 2
        return 18 * p["roughness_um"] + 240 * p["burr_mm"] + 8 * p["kerf_mm"] + 450 * p["energy_kwh_m"] + penalty
    r = differential_evolution(obj, bounds, seed=seed, maxiter=6, popsize=4, polish=True)
    return Recommendation(dict(zip(controls, map(float, r.x))), pred(r.x), float(r.fun))

if __name__ == "__main__":
    print(recommend_settings(fit_surrogates(generate_dataset())))
