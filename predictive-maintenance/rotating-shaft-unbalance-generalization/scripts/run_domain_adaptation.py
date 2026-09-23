from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import build_feature_table
from src.domain_adaptation import domain_adaptation_benchmark
from src.modeling import feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--representation", choices=["compact", "order"], default="compact")
    parser.add_argument("--adaptation-fraction", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-weight", type=float, default=8.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = ROOT / "data" / "processed" / "window_features.csv"
    if processed.exists():
        frame = pd.read_csv(processed)
    else:
        archive = ROOT / "data" / "raw" / cfg["source"]["archive_name"]
        if not archive.exists():
            raise FileNotFoundError(
                f"Missing {processed} and raw archive {archive}. Run scripts/download_data.py first."
            )
        frame = build_feature_table(
            archive,
            cfg["signal"],
            n_folds=int(cfg["validation"]["development_folds"]),
        )
        processed.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(processed, index=False)

    columns = feature_columns(frame, args.representation)
    result = domain_adaptation_benchmark(
        frame,
        columns,
        adaptation_fraction=args.adaptation_fraction,
        repeats=args.repeats,
        target_weight=args.target_weight,
        random_state=int(cfg["modeling"]["random_state"]),
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "domain_adaptation.csv"
    result.to_csv(output, index=False)
    summary = (
        result.groupby(["method", "k_per_class"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            accuracy=("accuracy", "mean"),
            severity_mae=("severity_mae", "mean"),
        )
    )
    print(summary.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
