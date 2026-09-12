from __future__ import annotations

import pandas as pd

from src.pareto import pareto_mask


def test_pareto_mask_minimizes_quality_load_and_maximizes_throughput():
    frame = pd.DataFrame({
        "ra": [1.0, 1.2, 0.8, 1.1],
        "force": [100, 120, 130, 90],
        "mrr": [10, 9, 8, 12],
    })
    mask = pareto_mask(frame, minimize=["ra", "force"], maximize=["mrr"])
    assert bool(mask[0])
    assert not bool(mask[1])
    assert bool(mask[2])
    assert bool(mask[3])
