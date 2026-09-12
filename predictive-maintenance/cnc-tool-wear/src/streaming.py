from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config, project_path
from .data import known_artifact_mask, load_experiment, mask_known_artifact_values
from .features import extract_window_features


def replay(experiment_path: str | Path, config_path: str | Path, emit_every: int) -> None:
    config = load_config(config_path)
    output_cfg = config["output"]
    artifact_dir = project_path(config, output_cfg["artifact_dir"])
    model = joblib.load(artifact_dir / output_cfg["model_file"])
    metadata = json.loads((artifact_dir / output_cfg["metadata_file"]).read_text(encoding="utf-8"))

    frame = load_experiment(experiment_path)
    window_size = int(metadata["window_size"])
    expected = list(metadata["feature_columns_numeric"]) + list(metadata["feature_columns_categorical"])
    threshold = float(metadata["probability_threshold"])
    sample_period = float(metadata["sample_period_seconds"])
    mask_artifacts = bool(metadata["mask_known_artifact_values"])
    buffer: deque[dict[str, object]] = deque(maxlen=window_size)

    for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
        one_row = pd.DataFrame([row])
        if mask_artifacts:
            one_row = mask_known_artifact_values(one_row)
        else:
            one_row["_known_artifact"] = known_artifact_mask(one_row).astype(int)
        buffer.append(one_row.iloc[0].to_dict())

        if len(buffer) < window_size or row_number % emit_every != 0:
            continue

        window = pd.DataFrame(list(buffer))
        features = extract_window_features(
            window,
            statistics=list(metadata["statistics"]),
            sample_period_seconds=sample_period,
        )
        feature_frame = pd.DataFrame([features])
        for column in expected:
            if column not in feature_frame.columns:
                feature_frame[column] = pd.NA
        probability = float(model.predict_proba(feature_frame[expected])[:, 1][0])
        payload = {
            "sample": row_number,
            "elapsed_seconds": round(row_number * sample_period, 3),
            "wear_probability": round(probability, 6),
            "alert": probability >= threshold,
        }
        print(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay CNC telemetry as an online condition-monitoring stream.")
    parser.add_argument("experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--emit-every", type=int, default=10)
    args = parser.parse_args()
    if args.emit_every <= 0:
        raise SystemExit("--emit-every must be positive")
    replay(args.experiment, args.config, args.emit_every)


if __name__ == "__main__":
    main()
