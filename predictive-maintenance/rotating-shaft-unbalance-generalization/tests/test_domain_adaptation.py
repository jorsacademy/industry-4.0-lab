import numpy as np
import pandas as pd

from src.domain_adaptation import (
    chronological_target_split,
    domain_adaptation_benchmark,
    fit_coral_transform,
)


def _fixture() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for session, shift in (("D", 0.0), ("E", 1.4)):
        for severity in range(5):
            values = rng.normal(
                loc=severity + shift,
                scale=0.7,
                size=(40, 4),
            )
            for window_index, row in enumerate(values):
                rows.append(
                    {
                        "recording": f"{severity}{session}",
                        "session": session,
                        "severity": severity,
                        "window_index": window_index,
                        **{f"f{i}": row[i] for i in range(4)},
                    }
                )
    return pd.DataFrame(rows)


def test_chronological_target_split_has_no_overlap() -> None:
    target = _fixture().query("session == 'E'").reset_index(drop=True)
    adaptation, evaluation = chronological_target_split(target, adaptation_fraction=0.25)
    assert len(np.intersect1d(adaptation, evaluation)) == 0
    assert len(adaptation) + len(evaluation) == len(target)
    for recording in target["recording"].unique():
        a = target.iloc[adaptation].query("recording == @recording")["window_index"]
        e = target.iloc[evaluation].query("recording == @recording")["window_index"]
        assert a.max() < e.min()


def test_coral_moves_source_covariance_toward_target_covariance() -> None:
    rng = np.random.default_rng(9)
    source = rng.normal(size=(300, 3))
    target = rng.normal(size=(300, 3)) @ np.diag([2.0, 0.5, 1.5]) + 2.0
    transform = fit_coral_transform(source, target)
    aligned = transform.transform_source(source)
    before = np.linalg.norm(np.cov(source, rowvar=False) - np.cov(target, rowvar=False))
    after = np.linalg.norm(np.cov(aligned, rowvar=False) - np.cov(target, rowvar=False))
    assert after < before


def test_domain_adaptation_benchmark_reports_all_methods() -> None:
    frame = _fixture()
    result = domain_adaptation_benchmark(
        frame,
        [f"f{i}" for i in range(4)],
        k_values=(1, 2),
        repeats=1,
        target_weight=4.0,
        random_state=4,
    )
    assert set(result["method"]) == {
        "source_only",
        "coral_unsupervised",
        "target_only_few_shot",
        "coral_plus_few_shot",
    }
    assert result["macro_f1"].between(0.0, 1.0).all()
