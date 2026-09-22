from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MeanEstimate:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True, slots=True)
class PairedEstimate:
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size_dz: float
    probability_of_superiority: float
    n_pairs: int


def _as_1d(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite numbers")
    return array


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    seed: int = 12_345,
) -> MeanEstimate:
    """Estimate a mean and percentile bootstrap confidence interval."""
    array = _as_1d(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if array.size == 1:
        return MeanEstimate(mean_value, std_value, mean_value, mean_value, 1)

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    chunk_size = min(1_000, n_bootstrap)
    offset = 0
    while offset < n_bootstrap:
        count = min(chunk_size, n_bootstrap - offset)
        indices = rng.integers(0, array.size, size=(count, array.size))
        bootstrap_means[offset : offset + count] = np.mean(array[indices], axis=1)
        offset += count

    ci_low, ci_high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return MeanEstimate(mean_value, std_value, float(ci_low), float(ci_high), int(array.size))


def paired_permutation_p_value(
    differences: list[float] | np.ndarray,
    *,
    n_permutations: int = 10_000,
    seed: int = 54_321,
) -> float:
    """Two-sided paired randomization test using sign flips."""
    diff = _as_1d(differences)
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    observed = abs(float(np.mean(diff)))
    if observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed = 0
    chunk_size = min(2_000, n_permutations)
    offset = 0
    while offset < n_permutations:
        count = min(chunk_size, n_permutations - offset)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, diff.size))
        permuted = np.abs(np.mean(signs * diff, axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        offset += count
    return float((exceed + 1) / (n_permutations + 1))


def paired_estimate(
    differences: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 5_000,
    n_permutations: int = 10_000,
    seed: int = 12_345,
) -> PairedEstimate:
    """Summarize paired improvements where positive values favor the candidate."""
    diff = _as_1d(differences)
    estimate = bootstrap_mean_ci(
        diff,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    p_value = paired_permutation_p_value(
        diff,
        n_permutations=n_permutations,
        seed=seed + 1,
    )
    if diff.size > 1:
        sd = float(np.std(diff, ddof=1))
        effect = (
            estimate.mean / sd
            if sd > 0.0
            else math.copysign(math.inf, estimate.mean)
            if estimate.mean
            else 0.0
        )
    else:
        effect = 0.0
    superiority = float(
        (np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)) / diff.size
    )
    return PairedEstimate(
        mean_difference=estimate.mean,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        p_value=p_value,
        effect_size_dz=float(effect),
        probability_of_superiority=superiority,
        n_pairs=int(diff.size),
    )
