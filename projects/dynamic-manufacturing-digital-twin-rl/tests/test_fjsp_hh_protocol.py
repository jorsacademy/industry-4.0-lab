from __future__ import annotations

from copy import deepcopy

import pytest

from dmdtrl import fjsp_hh_protocol
from dmdtrl.fjsp_operators import OPERATOR_NAMES

DESIGN_PATH = "configs/fjsp_hh_validation_design.json"


def _design() -> dict:
    return fjsp_hh_protocol.load_hh_validation_design(DESIGN_PATH)


def test_committed_design_is_valid_and_predeclared() -> None:
    design = _design()
    assert design["status"] == "predeclared_before_hyperheuristic_training"
    assert tuple(design["operator_names"]) == OPERATOR_NAMES
    assert design["training_seeds"] == [901, 1901, 2901, 3901, 4901]
    assert design["training_config"]["total_timesteps"] == 150_000
    assert design["validation_seed_start"] == 41_200
    assert design["validation_seed_end"] == 41_229
    assert design["final_test_seed_start"] == 42_000
    assert design["selection_data_boundary"]["final_test_used_for_selection"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("status",), "trained", "predeclared"),
        (("algorithm",), "DQN", "algorithm"),
        (("operator_names",), ["EARLIEST_DUE_DATE"], "operator_names"),
        (("training_seeds",), [901, 901, 2901, 3901, 4901], "unique"),
        (("training_config", "total_timesteps"), 149_999, "150000"),
        (("validation_seed_start",), 41_100, "overlap|41200"),
        (("validation_seed_count",), 29, "30 seeds"),
        (("final_test_seed_start",), 41_230, "100 seeds|42000|overlap"),
        (("frozen_cpsat_policy",), "OTHER", "CP-SAT"),
        (("selection_data_boundary", "final_test_used_for_selection"), True, "embargoed"),
        (("hyperparameter_selection", "search_performed_on_validation_block"), True, "forbidden"),
        (("inference_unit",), "episode", "training_seed"),
    ],
)
def test_protocol_rejects_post_hoc_or_boundary_changes(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    design = deepcopy(_design())
    target = design
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        fjsp_hh_protocol.validate_hh_validation_design(design)


def test_protocol_rejects_reversed_or_noninteger_seed_ranges() -> None:
    design = deepcopy(_design())
    design["development_seed_start"] = 41_000
    with pytest.raises(ValueError, match="reversed|overlap"):
        fjsp_hh_protocol.validate_hh_validation_design(design)

    design = deepcopy(_design())
    design["validation_seed_start"] = "41200"
    with pytest.raises(ValueError, match="integer"):
        fjsp_hh_protocol.validate_hh_validation_design(design)


def test_protocol_requires_five_nonnegative_training_seeds() -> None:
    design = deepcopy(_design())
    design["training_seeds"] = [901, 1901]
    with pytest.raises(ValueError, match="five"):
        fjsp_hh_protocol.validate_hh_validation_design(design)

    design = deepcopy(_design())
    design["training_seeds"] = [901, 1901, 2901, 3901, -1]
    with pytest.raises(ValueError, match="non-negative"):
        fjsp_hh_protocol.validate_hh_validation_design(design)
