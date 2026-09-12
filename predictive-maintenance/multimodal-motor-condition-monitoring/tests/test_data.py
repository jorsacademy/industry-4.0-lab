from pathlib import Path

import numpy as np
import pandas as pd

from src.data import blocked_purged_folds, load_frequency_table


def test_loader_finds_frequency_table_and_modalities(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for label in ["on", "unb"]:
        for i in range(10):
            rows.append({
                "label": label, "time": i * 0.04,
                "xAcc0010Hz": float(i), "yAcc0010Hz": float(i + 1), "zAcc0010Hz": float(i + 2),
                "snd0025Hz": float(i + 3),
            })
    pd.DataFrame(rows).to_csv(raw / "features.csv", index=False)
    table = load_frequency_table(raw)
    assert table.acceleration_columns == ["xAcc0010Hz", "yAcc0010Hz", "zAcc0010Hz"]
    assert table.audio_columns == ["snd0025Hz"]
    assert set(table.frame["condition"]) == {"on", "unb"}


def test_purged_folds_do_not_train_on_neighbor_windows():
    rows = []
    for label in ["on", "unb"]:
        for i in range(30):
            rows.append({"condition": label, "window_index": i})
    df = pd.DataFrame(rows)
    folds = blocked_purged_folds(df, n_splits=3, purge_windows=2)
    for train_idx, test_idx in folds:
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        for label in ["on", "unb"]:
            t = test.loc[test.condition == label, "window_index"].to_numpy()
            r = train.loc[train.condition == label, "window_index"].to_numpy()
            assert np.abs(r[:, None] - t[None, :]).min() > 2
