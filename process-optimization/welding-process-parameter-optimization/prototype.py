from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["current_a", "voltage_v", "travel_speed_mm_s", "wire_feed_m_min", "gas_flow_l_min", "thickness_mm"]
TARGETS = ["penetration_mm", "tensile_mpa", "porosity_risk", "energy_kj_mm"]
BOUNDS = {
    "current_a": (160.0, 300.0),
    "voltage_v": (20.0, 32.0),
    "travel_speed_mm_s": (3.0, 8.0),
    "wire_feed_m_min": (3.0, 9.0),
    "gas_flow_l_min": (10.0, 22.0),
    "thickness_mm": (2.0, 8.0),
}

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def simulate_process(X: pd.DataFrame, seed: int = 42, noisy: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = X.copy()
    heat = (x.voltage_v * x.current_a) / (1000.0 * x.travel_speed_mm_s)
    penetration = (
        0.55
        + 0.0072 * x.current_a
        + 0.055 * (x.voltage_v - 20)
        - 0.16 * (x.travel_speed_mm_s - 3)
        + 0.055 * (x.wire_feed_m_min - 3)
        - 0.08 * (x.thickness_mm - 4)
    )
    penetration = np.clip(penetration, 0.4, 7.5)
    pen_ratio = penetration / x.thickness_mm
    gas_penalty = ((x.gas_flow_l_min - 16.0) / 5.0) ** 2
    porosity = _sigmoid(-3.0 + 2.0 * gas_penalty + 0.35 * (x.travel_speed_mm_s - 5.0) - 0.8 * (x.wire_feed_m_min - 5.0))
    tensile = 470 + 250 * np.exp(-((pen_ratio - 0.72) / 0.20) ** 2) - 90 * porosity - 12 * np.maximum(heat - 2.6, 0)
    energy = heat
    if noisy:
        penetration = penetration + rng.normal(0, 0.08, len(x))
        tensile = tensile + rng.normal(0, 10.0, len(x))
        porosity = np.clip(porosity + rng.normal(0, 0.015, len(x)), 0, 1)
    return pd.DataFrame({
        "penetration_mm": penetration,
        "tensile_mpa": tensile,
        "porosity_risk": np.clip(porosity, 0, 1),
        "energy_kj_mm": energy,
    })

def generate_dataset(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({k: rng.uniform(*v, n) for k, v in BOUNDS.items()})
    y = simulate_process(X, seed=seed + 1, noisy=True)
    return pd.concat([X, y], axis=1)

def fit_surrogates(df: pd.DataFrame, seed: int = 42) -> dict[str, RandomForestRegressor]:
    models: dict[str, RandomForestRegressor] = {}
    for target in TARGETS:
        model = RandomForestRegressor(n_estimators=40, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        model.fit(df[FEATURES], df[target])
        models[target] = model
    return models

@dataclass(frozen=True)
class Recommendation:
    settings: dict[str, float]
    predicted: dict[str, float]
    objective: float

def recommend_settings(models: dict[str, RandomForestRegressor], thickness_mm: float = 5.0, seed: int = 42) -> Recommendation:
    controls = FEATURES[:-1]
    bounds = [BOUNDS[c] for c in controls]

    def predict(z: np.ndarray) -> dict[str, float]:
        row = pd.DataFrame([[*z, thickness_mm]], columns=FEATURES)
        return {t: float(models[t].predict(row)[0]) for t in TARGETS}

    def objective(z: np.ndarray) -> float:
        p = predict(z)
        min_pen, max_pen = 0.55 * thickness_mm, 0.92 * thickness_mm
        penalty = 250.0 * max(0.0, min_pen - p["penetration_mm"]) ** 2 + 250.0 * max(0.0, p["penetration_mm"] - max_pen) ** 2
        return -p["tensile_mpa"] + 180.0 * p["porosity_risk"] + 12.0 * p["energy_kj_mm"] + penalty

    result = differential_evolution(objective, bounds, seed=seed, polish=True, maxiter=6, popsize=4)
    pred = predict(result.x)
    return Recommendation(dict(zip(controls, map(float, result.x))), pred, float(result.fun))

def main() -> None:
    df = generate_dataset()
    models = fit_surrogates(df)
    rec = recommend_settings(models)
    print("Recommended welding settings")
    for k, v in rec.settings.items():
        print(f"  {k}: {v:.3f}")
    print("Predicted responses")
    for k, v in rec.predicted.items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
