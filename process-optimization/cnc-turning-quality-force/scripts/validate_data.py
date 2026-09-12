from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.data import aggregate_runs, available_force_columns, load_experiment


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    expected = cfg["validation"]
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]

    exp1 = load_experiment(raw_dir / data_cfg["exp1_file"], "Exp1")
    exp2 = load_experiment(raw_dir / data_cfg["exp2_file"], "Exp2")
    run1 = aggregate_runs(exp1)
    run2 = aggregate_runs(exp2)

    checks = {
        "exp1_rows": int(len(exp1)),
        "exp1_runs": int(len(run1)),
        "exp2_rows": int(len(exp2)),
        "exp2_runs": int(len(run2)),
        "exp1_position_counts": sorted(exp1.groupby("Run").size().unique().tolist()),
        "exp2_position_counts": sorted(exp2.groupby("Run").size().unique().tolist()),
        "exp2_tool_wear_levels": sorted(float(x) for x in exp2["TCond"].dropna().unique()),
        "force_columns": available_force_columns(run1),
    }
    print(checks)

    assert checks["exp1_rows"] == int(expected["exp1_rows"])
    assert checks["exp1_runs"] == int(expected["exp1_runs"])
    assert checks["exp2_rows"] == int(expected["exp2_rows"])
    assert checks["exp2_runs"] == int(expected["exp2_runs"])
    positions = int(expected["expected_positions_per_run"])
    assert checks["exp1_position_counts"] == [positions]
    assert checks["exp2_position_counts"] == [positions]
    assert np.allclose(checks["exp2_tool_wear_levels"], expected["exp2_tool_wear_levels"])
    assert "F" in checks["force_columns"]


if __name__ == "__main__":
    main()
