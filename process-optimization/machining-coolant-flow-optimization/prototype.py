from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import ExtraTreesRegressor

FEATURES = ["cutting_speed_m_min", "feed_mm_rev", "depth_mm", "coolant_flow_l_min", "coolant_temp_c", "concentration_pct"]
TARGETS = ["tool_temp_c", "roughness_um", "wear_rate", "pump_energy_index"]
BOUNDS = {
    "cutting_speed_m_min": (80.0, 300.0),
    "feed_mm_rev": (0.05, 0.30),
    "depth_mm": (0.5, 3.0),
    "coolant_flow_l_min": (2.0, 20.0),
    "coolant_temp_c": (15.0, 30.0),
    "concentration_pct": (3.0, 10.0),
}

def simulate_process(X, seed=42, noisy=True):
    rng = np.random.default_rng(seed)
    x = X.copy()
    load = (x.cutting_speed_m_min / 180) * (x.feed_mm_rev / 0.14) * (x.depth_mm / 1.5)
    cooling = (x.coolant_flow_l_min / 9) ** 0.65 * (25 / x.coolant_temp_c) ** 0.35 * (1 + 0.035 * (x.concentration_pct - 5))
    temp = 55 + 48 * load / (0.6 + cooling)
    rough = 0.35 + 1.15 * x.feed_mm_rev / 0.2 + 0.22 * x.depth_mm - 0.03 * x.coolant_flow_l_min + 0.012 * np.maximum(temp - 105, 0)
    wear = 0.018 + 0.035 * load * np.exp(np.maximum(temp - 90, 0) / 65)
    pump = (x.coolant_flow_l_min / 10) ** 2.2 * (0.7 + 0.03 * x.concentration_pct)
    if noisy:
        temp += rng.normal(0, 2.0, len(x))
        rough += rng.normal(0, 0.06, len(x))
        wear += rng.normal(0, 0.004, len(x))
    return pd.DataFrame({
        "tool_temp_c": np.clip(temp, 20, None),
        "roughness_um": np.clip(rough, 0.1, None),
        "wear_rate": np.clip(wear, 0, None),
        "pump_energy_index": pump,
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

def recommend_coolant(models, cutting_speed=220.0, feed=0.18, depth=2.0, max_tool_temp=120.0, seed=42):
    controls = ["coolant_flow_l_min", "coolant_temp_c", "concentration_pct"]
    bounds = [BOUNDS[c] for c in controls]
    def pred(z):
        row = pd.DataFrame([[cutting_speed, feed, depth, *z]], columns=FEATURES)
        return {t: float(models[t].predict(row)[0]) for t in TARGETS}
    def obj(z):
        p = pred(z)
        pen = 30 * max(0, p["tool_temp_c"] - max_tool_temp) ** 2
        return 1.2 * p["tool_temp_c"] + 55 * p["roughness_um"] + 900 * p["wear_rate"] + 22 * p["pump_energy_index"] + pen
    r = differential_evolution(obj, bounds, seed=seed, maxiter=6, popsize=4, polish=True)
    return Recommendation(dict(zip(controls, map(float, r.x))), pred(r.x), float(r.fun))

if __name__ == "__main__":
    print(recommend_coolant(fit_surrogates(generate_dataset())))
