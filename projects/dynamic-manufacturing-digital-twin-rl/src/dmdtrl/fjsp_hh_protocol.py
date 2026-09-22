from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dmdtrl.fjsp_operators import OPERATOR_NAMES


def load_hh_validation_design(path: str | Path) -> dict[str, Any]:
    design = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_hh_validation_design(design)
    return design


def _require_int(design: dict[str, Any], key: str) -> int:
    value = design.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def validate_hh_validation_design(design: dict[str, Any]) -> None:
    if design.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if design.get("phase") != 5:
        raise ValueError("phase must be 5")
    if design.get("status") != "predeclared_before_hyperheuristic_training":
        raise ValueError("validation design must be predeclared before training")
    if design.get("algorithm") != "PPO":
        raise ValueError("algorithm must be PPO")
    if design.get("controller") != "FJSP_HYPER_HEURISTIC":
        raise ValueError("controller must be FJSP_HYPER_HEURISTIC")
    if design.get("architecture") != "operator_selection_v1":
        raise ValueError("architecture must be operator_selection_v1")

    operators = tuple(design.get("operator_names", ()))
    if operators != OPERATOR_NAMES:
        raise ValueError("operator_names must match the frozen operator action space")
    if tuple(design.get("fixed_operator_baselines", ())) != OPERATOR_NAMES:
        raise ValueError("fixed_operator_baselines must contain every frozen operator")

    training_seeds = design.get("training_seeds")
    if not isinstance(training_seeds, list) or len(training_seeds) != 5:
        raise ValueError("exactly five training_seeds are required")
    if any(not isinstance(seed, int) or seed < 0 for seed in training_seeds):
        raise ValueError("training_seeds must be non-negative integers")
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training_seeds must be unique")

    training = design.get("training_config")
    if not isinstance(training, dict):
        raise ValueError("training_config must be an object")
    if training.get("total_timesteps") != 150_000:
        raise ValueError("total_timesteps must remain frozen at 150000")
    if training.get("n_steps") != 1_024 or training.get("batch_size") != 128:
        raise ValueError("PPO rollout and batch sizes must remain frozen")
    if training.get("device") != "cpu":
        raise ValueError("validation campaign device must remain cpu")

    development_start = _require_int(design, "development_seed_start")
    development_end = _require_int(design, "development_seed_end")
    or_start = _require_int(design, "or_tuning_seed_start")
    or_end = _require_int(design, "or_tuning_seed_end")
    direct_start = _require_int(design, "direct_action_validation_seed_start")
    direct_end = _require_int(design, "direct_action_validation_seed_end")
    validation_start = _require_int(design, "validation_seed_start")
    validation_count = _require_int(design, "validation_seed_count")
    validation_end = _require_int(design, "validation_seed_end")
    final_start = _require_int(design, "final_test_seed_start")
    final_count = _require_int(design, "final_test_seed_count")
    final_end = _require_int(design, "final_test_seed_end")

    ranges = [
        ("development", development_start, development_end),
        ("or_tuning", or_start, or_end),
        ("direct_validation", direct_start, direct_end),
        ("hyperheuristic_validation", validation_start, validation_end),
        ("final", final_start, final_end),
    ]
    for name, start, end in ranges:
        if start > end:
            raise ValueError(f"{name} seed range is reversed")
    for (_, _, left_end), (right_name, right_start, _) in zip(
        ranges,
        ranges[1:],
        strict=False,
    ):
        if left_end >= right_start:
            raise ValueError(f"seed ranges overlap before {right_name}")

    if validation_count != 30 or validation_end != validation_start + validation_count - 1:
        raise ValueError("hyper-heuristic validation block must contain exactly 30 seeds")
    if final_count != 100 or final_end != final_start + final_count - 1:
        raise ValueError("final block must contain exactly 100 seeds")
    if validation_start != 41_200 or validation_end != 41_229:
        raise ValueError("hyper-heuristic validation block must remain 41200-41229")
    if final_start != 42_000 or final_end != 42_099:
        raise ValueError("Phase-5 final block must remain 42000-42099")

    if design.get("frozen_cpsat_policy") != "FJSP_CPSAT_H4_B100MS":
        raise ValueError("frozen CP-SAT operating point changed")
    if design.get("frozen_cpsat_config_path") != "configs/fjsp_cpsat_validation_freeze.json":
        raise ValueError("frozen CP-SAT provenance path changed")

    selection = design.get("selection_data_boundary")
    if not isinstance(selection, dict):
        raise ValueError("selection_data_boundary must be an object")
    if selection.get("final_test_used_for_selection") is not False:
        raise ValueError("final test must remain embargoed during selection")

    hyperparameter_selection = design.get("hyperparameter_selection")
    if not isinstance(hyperparameter_selection, dict):
        raise ValueError("hyperparameter_selection must be an object")
    if hyperparameter_selection.get("search_performed_on_validation_block") is not False:
        raise ValueError("validation-block hyperparameter search is forbidden")
    if hyperparameter_selection.get("development_smoke_used_for_parameter_selection") is not False:
        raise ValueError("development smoke must not select hyperparameters")

    if design.get("inference_unit") != "training_seed":
        raise ValueError("top-level inference unit must be training_seed")
