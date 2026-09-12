from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import aggregate_runs, normalize_experiment


def test_run_aggregation_collapses_six_roughness_positions():
    rows = []
    for run, ap, ra0 in [(1, 0.5, 1.0), (2, 0.8, 2.0)]:
        for pos in range(6):
            rows.append({
                "Run": run,
                "Position": pos + 1,
                "ap": ap,
                "vc": 350,
                "f": 0.1,
                "TCond": 0.1,
                "Ra": ra0 + 0.1 * pos,
                "Fx": 100 + run,
                "Fy": 20 + run,
                "Fz": 30 + run,
                "F": 110 + run,
            })
    frame = normalize_experiment(pd.DataFrame(rows), "Exp2")
    run_level = aggregate_runs(frame)
    assert len(run_level) == 2
    assert run_level["n_positions"].tolist() == [6, 6]
    assert np.isclose(run_level.loc[0, "Ra_mean"], 1.25)
    assert run_level.loc[0, "condition_key"] != run_level.loc[1, "condition_key"]


def test_force_alias_fc_is_normalized_to_fx():
    frame = pd.DataFrame({
        "Run": [1], "ap": [0.5], "vc": [350], "f": [0.1], "Ra": [1.2],
        "Fc": [123.0], "Fy": [20.0], "Fz": [30.0], "F": [128.0],
    })
    out = normalize_experiment(frame, "Exp1")
    assert "Fx" in out.columns
    assert out.loc[0, "Fx"] == 123.0
    assert out.loc[0, "TCond"] == 0.0
