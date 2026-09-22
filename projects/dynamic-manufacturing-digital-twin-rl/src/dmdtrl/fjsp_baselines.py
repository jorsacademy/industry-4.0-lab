from __future__ import annotations

from collections.abc import Callable

from dmdtrl.fjsp_models import FJSPAction
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator

FJSPSelector = Callable[[FlexibleJobShopSimulator], FJSPAction]


def shortest_processing_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    actions = simulator.eligible_actions()
    if not actions:
        raise RuntimeError("no eligible FJSP action")

    def key(action: FJSPAction) -> tuple[float, float, int, int, int]:
        job = simulator.job(action.job_id)
        operation = job.operations[action.operation_index]
        machine = simulator.machines[action.machine_id]
        setup = simulator.setup_time(machine.last_family, job.family)
        return (
            setup + operation.processing_time_on(action.machine_id),
            job.due_date,
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(actions, key=key)


def earliest_due_date_action(simulator: FlexibleJobShopSimulator) -> FJSPAction:
    actions = simulator.eligible_actions()
    if not actions:
        raise RuntimeError("no eligible FJSP action")

    def key(action: FJSPAction) -> tuple[float, float, int, int, int]:
        job = simulator.job(action.job_id)
        operation = job.operations[action.operation_index]
        return (
            job.due_date,
            operation.processing_time_on(action.machine_id),
            action.job_id,
            action.operation_index,
            action.machine_id,
        )

    return min(actions, key=key)


def run_fjsp_policy(
    simulator: FlexibleJobShopSimulator,
    selector: FJSPSelector,
) -> dict[str, float]:
    while not simulator.terminated:
        simulator.step(selector(simulator))
    return simulator.metrics()
