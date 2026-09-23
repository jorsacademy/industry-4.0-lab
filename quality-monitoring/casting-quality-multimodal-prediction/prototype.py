from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROCESS_FEATURES = ["mold_temp_c", "pour_temp_c", "cooling_time_s", "pressure_bar", "alloy_index"]
VISION_FEATURES = ["image_porosity_score", "image_crack_score"]
FEATURES = PROCESS_FEATURES + VISION_FEATURES
BOUNDS = {
    "mold_temp_c": (160.0, 320.0),
    "pour_temp_c": (650.0, 780.0),
    "cooling_time_s": (25.0, 120.0),
    "pressure_bar": (20.0, 90.0),
    "alloy_index": (0.0, 1.0),
}

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_dataset(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({k: rng.uniform(*v, n) for k, v in BOUNDS.items()})
    process_latent = (
        np.abs(X.pour_temp_c - 715) / 34
        + np.abs(X.mold_temp_c - 235) / 62
        + np.abs(X.cooling_time_s - 68) / 42
        + np.maximum(45 - X.pressure_bar, 0) / 25
        + 0.8 * np.abs(X.alloy_index - 0.45)
    )
    true_prob = _sigmoid(-3.1 + 1.25 * process_latent)
    defect = rng.binomial(1, true_prob, n)
    X["image_porosity_score"] = np.clip(0.12 + 0.62 * defect + 0.16 * true_prob + rng.normal(0, 0.18, n), 0, 1)
    X["image_crack_score"] = np.clip(
        0.10 + 0.48 * defect + 0.20 * np.maximum((X.cooling_time_s - 80) / 40, 0) + rng.normal(0, 0.20, n),
        0,
        1,
    )
    X["defect"] = defect
    return X

@dataclass(frozen=True)
class Benchmark:
    process_auc: float
    vision_auc: float
    multimodal_auc: float

def benchmark(df: pd.DataFrame, seed=42) -> Benchmark:
    train, test = train_test_split(df, test_size=0.3, stratify=df.defect, random_state=seed)
    scores = []
    for cols in (PROCESS_FEATURES, VISION_FEATURES, FEATURES):
        model = RandomForestClassifier(
            n_estimators=60,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train[cols], train.defect)
        scores.append(roc_auc_score(test.defect, model.predict_proba(test[cols])[:, 1]))
    return Benchmark(*map(float, scores))

if __name__ == "__main__":
    print(benchmark(generate_dataset()))
