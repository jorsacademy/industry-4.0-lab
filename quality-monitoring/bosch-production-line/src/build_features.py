from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .data import iter_paired_chunks, raw_paths
from .features import FeatureOptions, extract_station_features


def _tag(prefix: float) -> str:
    return f"p{int(round(prefix * 100)):03d}"


def build(config_path: Path, prefix: float, max_rows: int | None = None) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    data_cfg = cfg["data"]
    feature_cfg = cfg.get("features", {})

    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    numeric_path, date_path = raw_paths(raw_dir, data_cfg["train_numeric"], data_cfg["train_date"])

    options = FeatureOptions(
        station_presence=bool(feature_cfg.get("include_station_presence", True)),
        station_time=bool(feature_cfg.get("include_station_time", True)),
        numeric_mean=bool(feature_cfg.get("include_station_numeric_mean", True)),
        numeric_std=bool(feature_cfg.get("include_station_numeric_std", True)),
        numeric_count=bool(feature_cfg.get("include_station_numeric_count", True)),
        line_station_counts=bool(feature_cfg.get("include_line_station_counts", True)),
    )

    output_path = processed_dir / f"bosch_features_{_tag(prefix)}.parquet"
    metadata_path = processed_dir / f"bosch_features_{_tag(prefix)}.json"
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows = 0
    positives = 0
    columns: list[str] = []
    try:
        for numeric_chunk, date_chunk in iter_paired_chunks(
            numeric_path,
            date_path,
            chunk_size=int(data_cfg.get("chunk_size", 5000)),
            max_rows=max_rows,
        ):
            features = extract_station_features(
                numeric_chunk,
                date_chunk,
                prefix_fraction=prefix,
                options=options,
            )
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                columns = list(features.columns)
            writer.write_table(table)
            rows += len(features)
            positives += int(features.get("Response", 0).sum())
    finally:
        if writer is not None:
            writer.close()

    if rows == 0:
        raise RuntimeError("No rows were produced")

    metadata = {
        "prefix_fraction": prefix,
        "rows": rows,
        "positive_responses": positives,
        "positive_rate": positives / rows if rows else None,
        "columns": columns,
        "raw_files": {
            "numeric": numeric_path.name,
            "date": date_path.name,
        },
        "max_rows": max_rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--prefix", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    build(args.config, args.prefix, args.max_rows)


if __name__ == "__main__":
    main()
