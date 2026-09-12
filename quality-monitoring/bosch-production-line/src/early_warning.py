from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .modeling import (
    PlattCalibrator,
    classification_metrics,
    feature_columns,
    fit_and_select_model,
    select_operating_threshold,
    temporal_four_way_split,
)


def evaluate_prefix(path: Path, cfg: dict) -> dict:
    frame = pd.read_parquet(path)
    split_cfg = cfg["split"]
    model_cfg = cfg["modeling"]
    split = temporal_four_way_split(
        frame,
        train_fraction=float(split_cfg["train_fraction"]),
        selection_fraction=float(split_cfg["selection_fraction"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )
    selected_name, model, _ = fit_and_select_model(
        frame,
        split,
        list(model_cfg.get("candidates", ["logistic", "hist_gradient"])),
        int(model_cfg.get("random_state", 42)),
    )
    columns = feature_columns(frame)
    x = frame[columns]
    y = frame["Response"].astype(int).to_numpy()
    raw_cal = model.predict_proba(x.iloc[split.calibration])[:, 1]
    calibrator = PlattCalibrator().fit(raw_cal, y[split.calibration])
    p_cal = calibrator.predict(raw_cal)
    threshold_cfg = model_cfg.get("operating_threshold", {})
    threshold = select_operating_threshold(
        y[split.calibration],
        p_cal,
        strategy=str(threshold_cfg.get("strategy", "min_precision")),
        min_precision=float(threshold_cfg.get("min_precision", 0.10)),
    )
    p_test = calibrator.predict(model.predict_proba(x.iloc[split.test])[:, 1])
    metrics = classification_metrics(y[split.test], p_test, threshold)
    tag = path.stem.rsplit("_", 1)[-1]
    prefix = int(tag[1:]) / 100.0 if tag.startswith("p") else float("nan")
    return {"prefix_fraction": prefix, "selected_model": selected_name, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--prefixes",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    processed = Path(cfg["data"]["processed_dir"])
    rows = []
    for prefix in args.prefixes:
        path = processed / f"bosch_features_p{int(round(prefix * 100)):03d}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Build it first with: python -m src.build_features --prefix {prefix}"
            )
        rows.append(evaluate_prefix(path, cfg))
    output = pd.DataFrame(rows).sort_values("prefix_fraction")
    reports_dir = Path(cfg["reports"].get("directory", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(reports_dir / "early_warning_tradeoff.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
