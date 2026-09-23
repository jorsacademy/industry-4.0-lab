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

from src.active_learning import benchmark_active_learning_strategies
from src.modeling import temporal_four_way_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "bosch_features_p100.parquet",
    )
    parser.add_argument("--initial-labels", type=int, default=2000)
    parser.add_argument("--query-batch", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--inspection-fraction", type=float, default=0.01)
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
    pool_idx = np.concatenate([split.train, split.selection, split.calibration])

    result = benchmark_active_learning_strategies(
        frame,
        pool_idx,
        split.test,
        initial_labels=args.initial_labels,
        query_batch_size=args.query_batch,
        rounds=args.rounds,
        candidate_pool_size=args.candidate_pool_size,
        inspection_fraction=args.inspection_fraction,
        random_state=int(cfg["modeling"].get("random_state", 42)),
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "active_learning_curve.csv"
    result.to_csv(output, index=False)
    print(result.groupby("strategy", as_index=False).tail(1).to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
