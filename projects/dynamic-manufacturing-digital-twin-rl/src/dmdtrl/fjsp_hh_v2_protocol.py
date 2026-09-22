from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dmdtrl.fjsp_operators import OPERATOR_NAMES


def load_hh_v2_boundary(path: str | Path) -> dict[str, Any]:
    boundary = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_hh_v2_boundary(boundary)
    return boundary


def _require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def validate_hh_v2_boundary(boundary: dict[str, Any]) -> None:
    if boundary.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if boundary.get("phase") != 5:
        raise ValueError("phase must be 5")
    if boundary.get("status") != "validation_block_reserved_before_v2_implementation":
        raise ValueError("v2 validation block must be reserved before implementation")
    if boundary.get("algorithm") != "PPO":
        raise ValueError("algorithm must remain PPO")
    if boundary.get("controller") != "FJSP_HYPER_HEURISTIC":
        raise ValueError("controller must remain FJSP_HYPER_HEURISTIC")
    if boundary.get("architecture") != "operator_selection_v2":
        raise ValueError("architecture must be operator_selection_v2")
    if boundary.get("predecessor_architecture") != "operator_selection_v1":
        raise ValueError("predecessor architecture must be operator_selection_v1")

    action_space = boundary.get("action_space")
    if not isinstance(action_space, dict):
        raise ValueError("action_space must be an object")
    if action_space.get("status") != "frozen_from_v1":
        raise ValueError("v2 action space must remain frozen from v1")
    if tuple(action_space.get("operator_names", ())) != OPERATOR_NAMES:
        raise ValueError("v2 operator action space must match the eight frozen operators")

    observation = boundary.get("observation_direction")
    if not isinstance(observation, dict):
        raise ValueError("observation_direction must be an object")
    if observation.get("family") != "global_plus_operator_conditioned_candidate_features":
        raise ValueError("unexpected v2 observation family")
    if observation.get("feature_schema_status") != "development_candidate_not_validation_frozen":
        raise ValueError("v2 feature schema must remain development-only at this stage")
    feature_families = observation.get("candidate_feature_families")
    if not isinstance(feature_families, list) or len(feature_families) < 5:
        raise ValueError("candidate_feature_families must contain the declared feature families")
    if len(set(feature_families)) != len(feature_families):
        raise ValueError("candidate_feature_families must be unique")

    seeds = boundary.get("seed_boundaries")
    if not isinstance(seeds, dict):
        raise ValueError("seed_boundaries must be an object")
    development_start = _require_int(seeds, "development_seed_start")
    development_end = _require_int(seeds, "development_seed_end")
    v1_start = _require_int(seeds, "consumed_v1_validation_seed_start")
    v1_end = _require_int(seeds, "consumed_v1_validation_seed_end")
    v2_start = _require_int(seeds, "v2_validation_seed_start")
    v2_end = _require_int(seeds, "v2_validation_seed_end")
    v2_count = _require_int(seeds, "v2_validation_seed_count")
    final_start = _require_int(seeds, "phase5_final_seed_start")
    final_end = _require_int(seeds, "phase5_final_seed_end")

    ranges = [
        ("development", development_start, development_end),
        ("consumed_v1_validation", v1_start, v1_end),
        ("reserved_v2_validation", v2_start, v2_end),
        ("phase5_final", final_start, final_end),
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

    if (development_start, development_end) != (40_000, 40_999):
        raise ValueError("development block must remain 40000-40999")
    if (v1_start, v1_end) != (41_200, 41_229):
        raise ValueError("consumed v1 validation block must remain 41200-41229")
    if v2_count != 30 or v2_end != v2_start + v2_count - 1:
        raise ValueError("v2 validation block must contain exactly 30 seeds")
    if (v2_start, v2_end) != (41_300, 41_329):
        raise ValueError("v2 validation block must remain 41300-41329")
    if (final_start, final_end) != (42_000, 42_099):
        raise ValueError("Phase-5 final block must remain 42000-42099")

    access = boundary.get("access_policy")
    if not isinstance(access, dict):
        raise ValueError("access_policy must be an object")
    if access.get("development_data_may_be_used_for_architecture_diagnostics") is not True:
        raise ValueError("development diagnostics must remain explicitly allowed")
    if access.get("v1_validation_may_be_used_only_as_historical_evidence") is not True:
        raise ValueError("v1 validation must remain historical evidence only")
    if access.get("v1_validation_may_be_reused_for_v2_selection") is not False:
        raise ValueError("consumed v1 validation cannot be reused for v2 selection")
    if access.get("v2_validation_access_authorized") is not False:
        raise ValueError("v2 validation access is not authorized by the boundary reservation")
    if access.get("phase5_final_access_authorized") is not False:
        raise ValueError("Phase-5 final access must remain unauthorized")

    requirements = boundary.get("future_validation_freeze_requirements")
    if not isinstance(requirements, dict):
        raise ValueError("future_validation_freeze_requirements must be an object")
    required_true = (
        "exact_observation_schema_must_be_frozen",
        "training_seeds_must_be_frozen",
        "ppo_hyperparameters_must_be_frozen",
        "baseline_set_must_include_all_eight_fixed_operators",
        "baseline_set_must_include_frozen_cpsat",
        "hyperparameter_search_on_v2_validation_forbidden",
        "validation_access_requires_separate_merged_protocol",
    )
    if any(requirements.get(key) is not True for key in required_true):
        raise ValueError("all future v2 validation freeze requirements must remain enabled")

    if boundary.get("frozen_cpsat_policy") != "FJSP_CPSAT_H4_B100MS":
        raise ValueError("frozen CP-SAT comparator changed")


def validate_against_v1_decision(
    boundary: dict[str, Any],
    v1_decision: dict[str, Any],
) -> None:
    validate_hh_v2_boundary(boundary)
    if v1_decision.get("architecture") != "operator_selection_v1":
        raise ValueError("v1 decision architecture mismatch")
    decision = v1_decision.get("decision")
    if not isinstance(decision, dict) or decision.get("promote_to_phase5_final") is not False:
        raise ValueError("v2 boundary requires a non-promoted v1 decision")
    previous_boundary = v1_decision.get("next_iteration_boundary")
    if not isinstance(previous_boundary, dict):
        raise ValueError("v1 decision is missing next-iteration boundary")
    if previous_boundary.get("may_reuse_41200_41229_for_selection") is not False:
        raise ValueError("v1 decision must prohibit validation reuse")
    if previous_boundary.get("requires_new_predeclared_validation_block") is not True:
        raise ValueError("v1 decision must require a new validation block")
    if previous_boundary.get("final_42000_42099_remains_embargoed") is not True:
        raise ValueError("v1 decision must preserve the final embargo")

    seeds = boundary["seed_boundaries"]
    v1_data = v1_decision.get("data_boundary")
    if not isinstance(v1_data, dict):
        raise ValueError("v1 decision is missing data boundary")
    if seeds["consumed_v1_validation_seed_start"] != v1_data.get("validation_seed_start"):
        raise ValueError("v2 boundary does not match consumed v1 validation start")
    if seeds["consumed_v1_validation_seed_end"] != v1_data.get("validation_seed_end"):
        raise ValueError("v2 boundary does not match consumed v1 validation end")
    if seeds["phase5_final_seed_start"] != v1_data.get("final_test_seed_start"):
        raise ValueError("v2 boundary changed Phase-5 final start")
    if seeds["phase5_final_seed_end"] != v1_data.get("final_test_seed_end"):
        raise ValueError("v2 boundary changed Phase-5 final end")
