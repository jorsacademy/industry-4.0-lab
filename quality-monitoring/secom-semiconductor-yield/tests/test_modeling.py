import numpy as np
import pandas as pd

from src.modeling import CandidateResult, build_models, choose_compact_candidate, choose_threshold, evaluate_candidate


def test_compact_candidate_prefers_small_panel_within_tolerance():
    model = build_models(1)["logistic_regression"]
    dummy = np.array([0.1, 0.9])
    results = [
        CandidateResult("a", 20, ["V0"], model, dummy, {"average_precision": 0.40, "brier_score": 0.2}),
        CandidateResult("b", 80, ["V0"], model, dummy, {"average_precision": 0.41, "brier_score": 0.1}),
        CandidateResult("c", 160, ["V0"], model, dummy, {"average_precision": 0.50, "brier_score": 0.1}),
    ]
    chosen = choose_compact_candidate(results, average_precision_tolerance=0.10)
    assert chosen.panel_size == 20


def test_threshold_meets_recall_constraint_when_possible():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.4, 0.6, 0.9])
    threshold = choose_threshold(y, p, min_recall=1.0)
    assert threshold <= 0.6


def test_candidate_training_smoke():
    rng = np.random.default_rng(7)
    train = pd.DataFrame({"V000": rng.normal(size=120), "V001": rng.normal(size=120), "is_fail": np.tile([0, 0, 0, 0, 0, 1], 20)})
    valid = train.sample(40, random_state=3).reset_index(drop=True)
    model = build_models(3)["logistic_regression"]
    result = evaluate_candidate(model, train, valid, ["V000", "V001"], "logistic")
    assert len(result.probabilities) == len(valid)
    assert np.isfinite(result.probabilities).all()
