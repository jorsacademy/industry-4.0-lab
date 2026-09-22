from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FJSPMachineOption:
    machine_id: int
    processing_time: float

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.processing_time <= 0.0:
            raise ValueError("processing_time must be positive")


@dataclass(slots=True, frozen=True)
class FJSPOperation:
    job_id: int
    operation_index: int
    machine_options: tuple[FJSPMachineOption, ...]

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("job_id must be non-negative")
        if self.operation_index < 0:
            raise ValueError("operation_index must be non-negative")
        if not self.machine_options:
            raise ValueError("an operation must have at least one eligible machine")
        machine_ids = [option.machine_id for option in self.machine_options]
        if len(machine_ids) != len(set(machine_ids)):
            raise ValueError("eligible machine ids must be unique within an operation")

    def processing_time_on(self, machine_id: int) -> float:
        for option in self.machine_options:
            if option.machine_id == machine_id:
                return option.processing_time
        raise ValueError(
            f"machine_id {machine_id} is not eligible for job {self.job_id} "
            f"operation {self.operation_index}"
        )

    @property
    def eligible_machine_ids(self) -> tuple[int, ...]:
        return tuple(option.machine_id for option in self.machine_options)


@dataclass(slots=True, frozen=True)
class FJSPJob:
    job_id: int
    release_time: float
    due_date: float
    priority: int
    family: int
    operations: tuple[FJSPOperation, ...]

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("job_id must be non-negative")
        if self.release_time < 0.0:
            raise ValueError("release_time must be non-negative")
        if self.due_date < self.release_time:
            raise ValueError("due_date must not precede release_time")
        if self.priority <= 0:
            raise ValueError("priority must be positive")
        if self.family < 0:
            raise ValueError("family must be non-negative")
        if not self.operations:
            raise ValueError("a job must contain at least one operation")
        expected = tuple(range(len(self.operations)))
        actual = tuple(operation.operation_index for operation in self.operations)
        if actual != expected:
            raise ValueError("operation indices must be consecutive and start at zero")
        if any(operation.job_id != self.job_id for operation in self.operations):
            raise ValueError("every operation must reference its parent job_id")


@dataclass(slots=True, frozen=True)
class FJSPInstance:
    jobs: tuple[FJSPJob, ...]
    n_machines: int

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("an instance must contain at least one job")
        if self.n_machines <= 0:
            raise ValueError("n_machines must be positive")
        job_ids = [job.job_id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("job ids must be unique")
        for job in self.jobs:
            for operation in job.operations:
                for machine_id in operation.eligible_machine_ids:
                    if machine_id >= self.n_machines:
                        raise ValueError(
                            f"machine_id {machine_id} is outside [0, {self.n_machines})"
                        )

    @property
    def total_operations(self) -> int:
        return sum(len(job.operations) for job in self.jobs)


@dataclass(slots=True)
class FJSPMachineState:
    machine_id: int
    available_at: float = 0.0
    last_family: int | None = None
    busy_time: float = 0.0
    setup_time: float = 0.0


@dataclass(slots=True, frozen=True, order=True)
class FJSPAction:
    job_id: int
    operation_index: int
    machine_id: int


@dataclass(slots=True, frozen=True)
class FJSPScheduledOperation:
    job_id: int
    operation_index: int
    machine_id: int
    family: int
    ready_time: float
    start_time: float
    completion_time: float
    processing_time: float
    setup_time: float
    waiting_time: float
