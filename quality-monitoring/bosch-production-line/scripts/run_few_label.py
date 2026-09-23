from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.few_label import few_label_adaptation_curve
from src.modeling import temporal_four_way_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "bosch_features_p100.parquet",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-weight", type=float, default=8.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    frame = pd.read_parquet(args.features)
    split_cfg = cfg["split"]
    split = temporal_four_way_split(
        frame,
        train_fraction=float(split_cfg["train_fraction"]),
        selection_fraction=float(split_cfg["selection_fraction"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )
    adaptation_idx = np.concatenate([split.selection, split.calibration])

    result = few_label_adaptation_curve(
        frame,
        split.train,
        adaptation_idx,
        split.test,
        repeats=args.repeats,
        target_weight=args.target_weight,
        random_state=int(cfg["modeling"].get("random_state", 42)),
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "few_label_adaptation.csv"
    result.to_csv(output, index=False)
    summary = (
        result.groupby(["method", "k_per_class"], as_index=False)
        .agg(
            average_precision=("average_precision", "mean"),
            f1=("f1", "mean"),
            recall=("recall", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
    )
    print(summary.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
