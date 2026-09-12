from __future__ import annotations

from pathlib import Path

import yaml

from src.data import load_electrical_report, measurement_columns


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    data_cfg = cfg["data"]
    expected = cfg["validation"]
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/download_data.py")

    df = load_electrical_report(
        path,
        target_column=data_cfg.get("target_column", "Result"),
        group_column=data_cfg.get("group_column", "LOT"),
        time_column=data_cfg.get("time_column", "Time"),
        pass_code=int(data_cfg.get("pass_code", 1)),
    )
    measurements = measurement_columns(df)
    checks = {
        "clean_rows": len(df),
        "lots": df["LOT"].nunique(),
        "defects": int(df["is_defect"].sum()),
        "measurements": len(measurements),
        "parsed_timestamps": int(df["timestamp"].notna().sum()),
    }
    print(checks)

    assert checks["measurements"] == int(data_cfg.get("expected_measurements", 19))
    assert checks["clean_rows"] == int(expected["expected_clean_rows"])
    assert checks["lots"] == int(expected["expected_lots"])
    assert checks["defects"] == int(expected["expected_defects"])


if __name__ == "__main__":
    main()
