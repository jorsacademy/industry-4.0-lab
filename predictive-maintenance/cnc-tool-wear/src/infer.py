from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config, project_path
from .data import clean_experiment, load_experiment
from .features import extract_window_features


def _feature_rows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    minimum_rows: int,
    statistics: list[str],
    sample_period_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start in range(0, max(1, len(frame) - window_size + 1), step_size):
        stop = min(start + window_size, len(frame))
        window = frame.iloc[start:stop]
        if len(window) < minimum_rows:
            continue
        features = extract_window_features(
            window,
            statistics=statistics,
            sample_period_seconds=sample_period_seconds,
        )
        rows.append(features)
    if not rows:
        raise ValueError("No valid inference windows were created from the supplied experiment.")
    return pd.DataFrame(rows)


def score_experiment(experiment_path: str | Path, config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    output_cfg = config["output"]
    artifact_dir = project_path(config, output_cfg["artifact_dir"])
    model = joblib.load(artifact_dir / output_cfg["model_file"])
    metadata = json.loads((artifact_dir / output_cfg["metadata_file"]).read_text(encoding="utf-8"))

    frame = clean_experiment(
        load_experiment(experiment_path),
        mask_known_artifacts=bool(metadata["mask_known_artifact_values"]),
    )
    features = _feature_rows(
        frame,
        window_size=int(metadata["window_size"]),
        step_size=int(metadata["step_size"]),
        minimum_rows=int(metadata["minimum_rows_per_window"]),
        statistics=list(metadata["statistics"]),
        sample_period_seconds=float(metadata["sample_period_seconds"]),
    )

    expected = list(metadata["feature_columns_numeric"]) + list(metadata["feature_columns_categorical"])
    for column in expected:
        if column not in features.columns:
            features[column] = pd.NA
    probabilities = model.predict_proba(features[expected])[:, 1]
    threshold = float(metadata["probability_threshold"])
    mean_probability = float(probabilities.mean())
    return {
        "windows": int(len(probabilities)),
        "mean_probability": mean_probability,
        "median_probability": float(pd.Series(probabilities).median()),
        "latest_probability": float(probabilities[-1]),
        "threshold": threshold,
        "decision": "worn" if mean_probability >= threshold else "unworn",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a complete CNC experiment.")
    parser.add_argument("experiment")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(score_experiment(args.experiment, args.config), indent=2))


if __name__ == "__main__":
    main()
