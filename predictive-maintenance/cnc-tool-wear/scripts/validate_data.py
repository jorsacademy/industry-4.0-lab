from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import load_config, project_path
from src.data import validate_dataset


def main() -> None:
    config = load_config(PROJECT_DIR / "config.yaml")
    data_cfg = config["data"]
    summary = validate_dataset(
        project_path(config, data_cfg["raw_dir"]),
        metadata_file=data_cfg["metadata_file"],
        pattern=data_cfg["experiment_glob"],
        expected_experiments=int(data_cfg["expected_experiments"]),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
