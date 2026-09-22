import numpy as np
import pytest

from dmdtrl.statistics import bootstrap_mean_ci, paired_estimate, paired_permutation_p_value


def test_bootstrap_mean_ci_is_reproducible_and_contains_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = bootstrap_mean_ci(values, n_bootstrap=1_000, seed=7)
    second = bootstrap_mean_ci(values, n_bootstrap=1_000, seed=7)
    assert first == second
    assert first.mean == pytest.approx(3.0)
    assert first.ci_low <= first.mean <= first.ci_high
    assert first.n == 5


def test_paired_permutation_detects_clear_direction():
    differences = np.arange(1.0, 13.0)
    p_value = paired_permutation_p_value(differences, n_permutations=4_000, seed=3)
    assert p_value < 0.01


def test_paired_estimate_reports_probability_of_superiority():
    estimate = paired_estimate([1.0, 2.0, 0.0, -1.0], n_bootstrap=500, n_permutations=500)
    assert estimate.n_pairs == 4
    assert estimate.probability_of_superiority == pytest.approx(0.625)


def test_statistics_reject_invalid_input():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, np.nan])
