from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Job:
    job_id: int
    arrival_time: float
    processing_time: float
    due_date: float
    priority: int
    family: int
    quality_risk: float


@dataclass(slots=True)
class Machine:
    machine_id: int
    speed: float
    available_at: float = 0.0
    last_family: int | None = None
    busy_time: float = 0.0
    setup_time: float = 0.0
    repair_time: float = 0.0


@dataclass(slots=True, frozen=True)
class ScheduledOperation:
    job_id: int
    machine_id: int
    family: int
    start_time: float
    completion_time: float
    processing_time: float
    setup_time: float
    repair_time: float
    waiting_time: float
    tardiness: float
    weighted_tardiness: float
    on_time: bool
    rule: str
