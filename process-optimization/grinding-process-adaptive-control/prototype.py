from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["wheel_speed_m_s", "work_speed_m_min", "depth_mm", "feed_mm_rev", "coolant_l_min", "wheel_age"]
TARGETS = ["roughness_um", "burn_risk", "spindle_power_kw", "wear_rate"]
BOUNDS = {
    "wheel_speed_m_s": (20.0, 50.0),
    "work_speed_m_min": (5.0, 25.0),
    "depth_mm": (0.005, 0.05),
    "feed_mm_rev": (0.2, 1.5),
    "coolant_l_min": (2.0, 12.0),
    "wheel_age": (0.0, 1.0),
}

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_process(X, seed=42, noisy=True):
    rng = np.random.default_rng(seed)
    x = X.copy()
    load = (x.depth_mm / 0.02) * (x.feed_mm_rev / 0.6) * (x.work_speed_m_min / 12)
    cooling = x.coolant_l_min / 7
    rough = 0.35 + 0.5 * load + 1.4 * x.wheel_age - 0.018 * x.wheel_speed_m_s - 0.11 * cooling
    thermal = 1.2 * load + 1.6 * x.wheel_age - 0.8 * cooling - 0.015 * (x.wheel_speed_m_s - 30)
    burn = _sigmoid(thermal - 1.4)
    power = 1.5 + 2.4 * load + 0.9 * x.wheel_age + 0.025 * x.wheel_speed_m_s
    wear = 0.04 + 0.11 * load + 0.16 * x.wheel_age + 0.025 * np.maximum(x.wheel_speed_m_s - 42, 0)
    if noisy:
        rough += rng.normal(0, 0.05, len(x))
        burn = np.clip(burn + rng.normal(0, 0.012, len(x)), 0, 1)
        power += rng.normal(0, 0.07, len(x))
    return pd.DataFrame({
        "roughness_um": np.clip(rough, 0.1, None),
        "burn_risk": burn,
        "spindle_power_kw": np.clip(power, 0.1, None),
        "wear_rate": np.clip(wear, 0, None),
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

def recommend_settings(models, wheel_age=0.65, seed=42):
    controls = FEATURES[:-1]
    bounds = [BOUNDS[c] for c in controls]
    def pred(z):
        row = pd.DataFrame([[*z, wheel_age]], columns=FEATURES)
        return {t: float(models[t].predict(row)[0]) for t in TARGETS}
    def obj(z):
        p = pred(z)
        pen = 1200 * max(0, p["burn_risk"] - 0.20) ** 2 + 80 * max(0, p["roughness_um"] - 1.2) ** 2
        productivity = z[1] * z[2] * z[3]
        return 45 * p["roughness_um"] + 180 * p["burn_risk"] + 5 * p["spindle_power_kw"] + 80 * p["wear_rate"] - 12 * productivity + pen
    r = differential_evolution(obj, bounds, seed=seed, maxiter=6, popsize=4, polish=True)
    return Recommendation(dict(zip(controls, map(float, r.x))), pred(r.x), float(r.fun))

if __name__ == "__main__":
    print(recommend_settings(fit_surrogates(generate_dataset())))
