from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from dmdtrl.fjsp_models import (
    FJSPAction,
    FJSPInstance,
    FJSPJob,
    FJSPMachineState,
    FJSPScheduledOperation,
)

_EPS = 1e-12


class FlexibleJobShopSimulator:
    """Deterministic event-driven FJSP scheduling core.

    Jobs contain ordered operations. Each operation has its own eligible-machine
    set and machine-dependent processing time. A decision assigns only the next
    precedence-feasible operation of a released job to a currently available
    eligible machine.
    """

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        default_setup_time: float = 0.0,
        setup_times: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        if default_setup_time < 0.0:
            raise ValueError("default_setup_time must be non-negative")
        if setup_times and any(value < 0.0 for value in setup_times.values()):
            raise ValueError("setup times must be non-negative")
        self.instance = instance
        self.default_setup_time = float(default_setup_time)
        self.setup_times = dict(setup_times or {})
        self._jobs = {job.job_id: job for job in instance.jobs}
        self.reset()

    def reset(self) -> None:
        self.current_time = 0.0
        self.machines = {
            machine_id: FJSPMachineState(machine_id=machine_id)
            for machine_id in range(self.instance.n_machines)
        }
        self.next_operation = {job.job_id: 0 for job in self.instance.jobs}
        self.operation_ready_at = {
            job.job_id: job.release_time for job in self.instance.jobs
        }
        self.job_completion_times: dict[int, float] = {}
        self.schedule: list[FJSPScheduledOperation] = []
        self._advance_to_decision()

    @property
    def terminated(self) -> bool:
        return len(self.schedule) == self.instance.total_operations

    def job(self, job_id: int) -> FJSPJob:
        return self._jobs[int(job_id)]

    def eligible_actions(self) -> tuple[FJSPAction, ...]:
        if self.terminated:
            return ()
        actions: list[FJSPAction] = []
        for job in sorted(self.instance.jobs, key=lambda item: item.job_id):
            operation_index = self.next_operation[job.job_id]
            if operation_index >= len(job.operations):
                continue
            if self.operation_ready_at[job.job_id] > self.current_time + _EPS:
                continue
            operation = job.operations[operation_index]
            for machine_id in operation.eligible_machine_ids:
                machine = self.machines[machine_id]
                if machine.available_at <= self.current_time + _EPS:
                    actions.append(
                        FJSPAction(
                            job_id=job.job_id,
                            operation_index=operation_index,
                            machine_id=machine_id,
                        )
                    )
        return tuple(sorted(actions))

    def step(self, action: FJSPAction) -> bool:
        if self.terminated:
            raise RuntimeError("schedule is already complete")
        eligible = self.eligible_actions()
        if action not in eligible:
            raise ValueError(f"action {action} is not precedence/resource feasible")

        job = self._jobs[action.job_id]
        operation = job.operations[action.operation_index]
        machine = self.machines[action.machine_id]
        ready_time = self.operation_ready_at[job.job_id]
        start_time = max(self.current_time, ready_time, machine.available_at)
        setup_time = self.setup_time(machine.last_family, job.family)
        processing_time = operation.processing_time_on(action.machine_id)
        completion_time = start_time + setup_time + processing_time
        waiting_time = max(0.0, start_time - ready_time)

        self.schedule.append(
            FJSPScheduledOperation(
                job_id=job.job_id,
                operation_index=operation.operation_index,
                machine_id=machine.machine_id,
                family=job.family,
                ready_time=ready_time,
                start_time=start_time,
                completion_time=completion_time,
                processing_time=processing_time,
                setup_time=setup_time,
                waiting_time=waiting_time,
            )
        )
        machine.available_at = completion_time
        machine.last_family = job.family
        machine.busy_time += processing_time
        machine.setup_time += setup_time

        next_index = action.operation_index + 1
        self.next_operation[job.job_id] = next_index
        self.operation_ready_at[job.job_id] = completion_time
        if next_index == len(job.operations):
            self.job_completion_times[job.job_id] = completion_time

        if self.terminated:
            self.current_time = self.metrics()["makespan"]
        else:
            self._advance_to_decision()
        return self.terminated

    def step_assignment(self, job_id: int, operation_index: int, machine_id: int) -> bool:
        return self.step(
            FJSPAction(
                job_id=int(job_id),
                operation_index=int(operation_index),
                machine_id=int(machine_id),
            )
        )

    def setup_time(self, previous_family: int | None, next_family: int) -> float:
        if previous_family is None or previous_family == next_family:
            return 0.0
        return float(
            self.setup_times.get(
                (previous_family, next_family),
                self.default_setup_time,
            )
        )

    def _advance_to_decision(self) -> None:
        while not self.terminated:
            if self.eligible_actions():
                return

            candidates: list[float] = []
            for job in self.instance.jobs:
                operation_index = self.next_operation[job.job_id]
                if operation_index >= len(job.operations):
                    continue
                ready_time = self.operation_ready_at[job.job_id]
                if ready_time > self.current_time + _EPS:
                    candidates.append(ready_time)
            candidates.extend(
                machine.available_at
                for machine in self.machines.values()
                if machine.available_at > self.current_time + _EPS
            )
            if not candidates:
                raise RuntimeError("FJSP state is deadlocked with unscheduled operations")
            self.current_time = min(candidates)

    def metrics(self) -> dict[str, float]:
        makespan = max(
            (operation.completion_time for operation in self.schedule),
            default=0.0,
        )
        total_busy = sum(machine.busy_time for machine in self.machines.values())
        total_setup = sum(machine.setup_time for machine in self.machines.values())
        tardiness = 0.0
        weighted_tardiness = 0.0
        flow_time = 0.0
        for job_id, completion_time in self.job_completion_times.items():
            job = self._jobs[job_id]
            job_tardiness = max(0.0, completion_time - job.due_date)
            tardiness += job_tardiness
            weighted_tardiness += job.priority * job_tardiness
            flow_time += completion_time - job.release_time

        completed_jobs = len(self.job_completion_times)
        utilization = total_busy / max(self.instance.n_machines * makespan, _EPS)
        mean_operation_waiting = sum(op.waiting_time for op in self.schedule) / max(
            len(self.schedule), 1
        )
        return {
            "scheduled_operations": float(len(self.schedule)),
            "completed_jobs": float(completed_jobs),
            "makespan": float(makespan),
            "total_tardiness": float(tardiness),
            "weighted_tardiness": float(weighted_tardiness),
            "mean_flow_time": float(flow_time / max(completed_jobs, 1)),
            "mean_operation_waiting_time": float(mean_operation_waiting),
            "total_setup_time": float(total_setup),
            "utilization": float(utilization),
        }

    def schedule_records(self) -> list[dict[str, float | int]]:
        return [asdict(operation) for operation in self.schedule]
