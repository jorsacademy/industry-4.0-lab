from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from dmdtrl.fjsp_hh_v2_protocol import (
    load_hh_v2_boundary,
    validate_against_v1_decision,
    validate_hh_v2_boundary,
)
from dmdtrl.fjsp_operators import OPERATOR_NAMES

BOUNDARY_PATH = Path("configs/fjsp_hh_v2_boundary.json")
V1_DECISION_PATH = Path("configs/fjsp_hh_validation_decision.json")


def _boundary() -> dict:
    return load_hh_v2_boundary(BOUNDARY_PATH)


def _v1_decision() -> dict:
    return json.loads(V1_DECISION_PATH.read_text(encoding="utf-8"))


def test_committed_v2_boundary_is_valid_and_reserved_before_implementation() -> None:
    boundary = _boundary()
    assert boundary["status"] == "validation_block_reserved_before_v2_implementation"
    assert boundary["architecture"] == "operator_selection_v2"
    assert tuple(boundary["action_space"]["operator_names"]) == OPERATOR_NAMES
    seeds = boundary["seed_boundaries"]
    assert seeds["v2_validation_seed_start"] == 41_300
    assert seeds["v2_validation_seed_end"] == 41_329
    assert seeds["phase5_final_seed_start"] == 42_000
    assert boundary["access_policy"]["v2_validation_access_authorized"] is False
    assert boundary["access_policy"]["phase5_final_access_authorized"] is False


def test_v2_boundary_matches_frozen_v1_rejection_decision() -> None:
    validate_against_v1_decision(_boundary(), _v1_decision())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("architecture",), "operator_selection_v1", "operator_selection_v2"),
        (("action_space", "status"), "changed", "frozen"),
        (("action_space", "operator_names"), ["EARLIEST_DUE_DATE"], "eight frozen"),
        (
            ("observation_direction", "feature_schema_status"),
            "validation_frozen",
            "development-only",
        ),
        (("seed_boundaries", "v2_validation_seed_start"), 41_200, "overlap|41300"),
        (("seed_boundaries", "v2_validation_seed_count"), 29, "30 seeds"),
        (("seed_boundaries", "phase5_final_seed_start"), 41_330, "42000|100 seeds"),
        (("access_policy", "v1_validation_may_be_reused_for_v2_selection"), True, "cannot"),
        (("access_policy", "v2_validation_access_authorized"), True, "not authorized"),
        (("access_policy", "phase5_final_access_authorized"), True, "unauthorized"),
        (("frozen_cpsat_policy",), "OTHER", "CP-SAT"),
    ],
)
def test_boundary_rejects_leakage_or_post_hoc_changes(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    boundary = deepcopy(_boundary())
    target = boundary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_hh_v2_boundary(boundary)


def test_boundary_rejects_duplicate_feature_families() -> None:
    boundary = deepcopy(_boundary())
    features = boundary["observation_direction"]["candidate_feature_families"]
    features[-1] = features[0]
    with pytest.raises(ValueError, match="unique"):
        validate_hh_v2_boundary(boundary)


def test_cross_validation_rejects_v1_reuse_or_final_boundary_drift() -> None:
    decision = deepcopy(_v1_decision())
    decision["next_iteration_boundary"]["may_reuse_41200_41229_for_selection"] = True
    with pytest.raises(ValueError, match="prohibit"):
        validate_against_v1_decision(_boundary(), decision)

    boundary = deepcopy(_boundary())
    boundary["seed_boundaries"]["consumed_v1_validation_seed_end"] = 41_228
    with pytest.raises(ValueError, match="41200-41229|consumed v1"):
        validate_against_v1_decision(boundary, _v1_decision())


def test_future_validation_freeze_requirements_cannot_be_disabled() -> None:
    boundary = deepcopy(_boundary())
    boundary["future_validation_freeze_requirements"][
        "exact_observation_schema_must_be_frozen"
    ] = False
    with pytest.raises(ValueError, match="requirements"):
        validate_hh_v2_boundary(boundary)
