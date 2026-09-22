"""Interpretable maintenance policies used as industrial baselines."""

from __future__ import annotations

import numpy as np


def run_to_failure_policy(observation: np.ndarray) -> int:
    del observation
    return 0


def age_based_policy(observation: np.ndarray, age_threshold: float = 0.45) -> int:
    normalized_age = float(observation[1])
    return 2 if normalized_age >= age_threshold else 0


def condition_based_policy(
    observation: np.ndarray,
    maintenance_threshold: float = 0.65,
    replacement_threshold: float = 0.90,
) -> int:
    degradation_signal = float(observation[3])
    if degradation_signal >= replacement_threshold:
        return 3
    if degradation_signal >= maintenance_threshold:
        return 2
    return 0
