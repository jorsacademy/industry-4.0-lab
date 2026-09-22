from __future__ import annotations

from enum import IntEnum

from dmdtrl.fjsp_baselines import earliest_due_date_action, shortest_processing_action
from dmdtrl.fjsp_models import FJSPAction
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator

_EPS = 1e-12


class FJSPOperator(IntEnum):
    """Stable action IDs for the FJSP hyper-heuristic controller."""

    EARLIEST_DUE_DATE = 0
    SHORTEST_PROCESSING = 1
    MINIMUM_SETUP = 2
    HIGHEST_PRIORITY = 3
    MINIMUM_SLACK = 4
    CRITICAL_RATIO = 5
    SAME_FAMILY_FIRST = 6
    WEIGHTED_TARDINESS_RISK = 7


OPERATOR_NAMES = tuple(operator.name for operator in FJSPOperator)


def _eligible_actions(simulator: FlexibleJobShopSimulator) -> tuple[FJSPAction, ...]:
    actions = simulator.eligible_actions()
    if not actions:
        raise RuntimeError("no eligible FJSP action")
    return actions


def _processing_and_setup(
    simulator: FlexibleJobShopSimulator,
    action: FJSPAction,
) -> tuple[float, float]:
    job = simulator.job(action.job_id)
    operation = job.operations[action.operation_index]
    machine = simulator.machines[action.machine_id]
    processing = operation.processing_time_on(action.machine_id)
    setup = simulator.setup_time(machine.last_family, job.family)
    return processing, setup


def _remaining_min_processing(
    simulator: FlexibleJobShopSimulator,
    job_id: int,
    *,
    after_current: bool = False,
) -> float:
    job = simulator.job(job_id)
    next_index = simulator.next_operation[job_id]
    start = next_index + (1 if after_current else 0)
    return float(
        sum(
            min(option.processing_time for option in operation.machine_options)
            for operation in job.operations[start:]
        )
    )


def minimum_setup_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    def key(action: FJSPAction) -> tuple[float, float, float, int, int, int]:
        job = simulator.job(action.job_id)
        processing, setup = _processing_and_setup(simulator, action)
        return (
            setup,
            processing,
            job.due_date,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(_eligible_actions(simulator), key=key)


def highest_priority_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    def key(action: FJSPAction) -> tuple[float, float, float, int, int, int]:
        job = simulator.job(action.job_id)
        processing, setup = _processing_and_setup(simulator, action)
        return (
            -job.priority,
            job.due_date,
            setup + processing,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(_eligible_actions(simulator), key=key)


def minimum_slack_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    def key(action: FJSPAction) -> tuple[float, float, float, int, int, int]:
        job = simulator.job(action.job_id)
        processing, setup = _processing_and_setup(simulator, action)
        remaining = _remaining_min_processing(simulator, action.job_id)
        slack = job.due_date - simulator.current_time - remaining
        return (
            slack,
            -job.priority,
            setup + processing,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(_eligible_actions(simulator), key=key)


def critical_ratio_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    def key(action: FJSPAction) -> tuple[float, float, float, int, int, int]:
        job = simulator.job(action.job_id)
        processing, setup = _processing_and_setup(simulator, action)
        remaining = max(_remaining_min_processing(simulator, action.job_id), _EPS)
        ratio = (job.due_date - simulator.current_time) / remaining
        return (
            ratio,
            -job.priority,
            setup + processing,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(_eligible_actions(simulator), key=key)


def same_family_first_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    def key(action: FJSPAction) -> tuple[int, float, float, float, int, int, int]:
        job = simulator.job(action.job_id)
        machine = simulator.machines[action.machine_id]
        processing, setup = _processing_and_setup(simulator, action)
        family_mismatch = int(machine.last_family != job.family)
        return (
            family_mismatch,
            setup,
            processing,
            job.due_date,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(_eligible_actions(simulator), key=key)


def weighted_tardiness_risk_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    """Prioritize the largest lower-bound weighted tardiness exposure."""

    def key(action: FJSPAction) -> tuple[float, float, float, int, int, int]:
        job = simulator.job(action.job_id)
        processing, setup = _processing_and_setup(simulator, action)
        remaining_after = _remaining_min_processing(
            simulator,
            action.job_id,
            after_current=True,
        )
        projected_completion = (
            simulator.current_time + setup + processing + remaining_after
        )
        risk = job.priority * max(0.0, projected_completion - job.due_date)
        slack = job.due_date - simulator.current_time - processing - remaining_after
        return (
            -risk,
            slack,
            setup + processing,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(_eligible_actions(simulator), key=key)


def select_operator_action(
    simulator: FlexibleJobShopSimulator,
    operator: FJSPOperator | int,
) -> FJSPAction:
    """Map one small discrete operator action to a feasible simulator action."""

    selected = FJSPOperator(int(operator))
    if selected is FJSPOperator.EARLIEST_DUE_DATE:
        return earliest_due_date_action(simulator)
    if selected is FJSPOperator.SHORTEST_PROCESSING:
        return shortest_processing_action(simulator)
    if selected is FJSPOperator.MINIMUM_SETUP:
        return minimum_setup_action(simulator)
    if selected is FJSPOperator.HIGHEST_PRIORITY:
        return highest_priority_action(simulator)
    if selected is FJSPOperator.MINIMUM_SLACK:
        return minimum_slack_action(simulator)
    if selected is FJSPOperator.CRITICAL_RATIO:
        return critical_ratio_action(simulator)
    if selected is FJSPOperator.SAME_FAMILY_FIRST:
        return same_family_first_action(simulator)
    if selected is FJSPOperator.WEIGHTED_TARDINESS_RISK:
        return weighted_tardiness_risk_action(simulator)
    raise AssertionError(f"unhandled FJSP operator {selected}")
