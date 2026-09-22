from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from dmdtrl.models import Job, Machine


class DispatchRule(IntEnum):
    FIFO = 0
    HIGHEST_PRIORITY = 1
    EARLIEST_DUE_DATE = 2
    SHORTEST_PROCESSING_TIME = 3
    SAME_FAMILY_FIRST = 4
    MINIMUM_SETUP = 5
    CRITICAL_RATIO = 6
    WEIGHTED_COMPOSITE = 7


RULE_NAMES = tuple(rule.name.lower() for rule in DispatchRule)


def _setup(job: Job, machine: Machine, setup_time: float) -> float:
    if machine.last_family is None or machine.last_family == job.family:
        return 0.0
    return setup_time


def select_job(
    queue: Sequence[Job],
    machine: Machine,
    current_time: float,
    rule: DispatchRule,
    setup_time: float,
) -> Job:
    """Select one feasible queued job under a dispatching rule."""
    if not queue:
        raise ValueError("queue must contain at least one job")

    if rule is DispatchRule.FIFO:
        return min(queue, key=lambda j: (j.arrival_time, j.job_id))
    if rule is DispatchRule.HIGHEST_PRIORITY:
        return min(queue, key=lambda j: (-j.priority, j.due_date, j.job_id))
    if rule is DispatchRule.EARLIEST_DUE_DATE:
        return min(queue, key=lambda j: (j.due_date, -j.priority, j.job_id))
    if rule is DispatchRule.SHORTEST_PROCESSING_TIME:
        return min(
            queue,
            key=lambda j: (j.processing_time / machine.speed, j.due_date, j.job_id),
        )
    if rule is DispatchRule.SAME_FAMILY_FIRST:
        return min(
            queue,
            key=lambda j: (
                0 if machine.last_family == j.family else 1,
                j.due_date,
                j.job_id,
            ),
        )
    if rule is DispatchRule.MINIMUM_SETUP:
        return min(queue, key=lambda j: (_setup(j, machine, setup_time), j.due_date, j.job_id))
    if rule is DispatchRule.CRITICAL_RATIO:
        return min(
            queue,
            key=lambda j: (
                (j.due_date - current_time) / max(j.processing_time / machine.speed, 1e-9),
                -j.priority,
                j.job_id,
            ),
        )
    if rule is DispatchRule.WEIGHTED_COMPOSITE:
        max_proc = max(j.processing_time for j in queue)
        max_due = max(max(j.due_date - current_time, 0.0) for j in queue) or 1.0

        def composite_key(j: Job) -> tuple[float, int]:
            proc = (j.processing_time / machine.speed) / max(max_proc, 1e-9)
            due_pressure = max(j.due_date - current_time, 0.0) / max_due
            setup = _setup(j, machine, setup_time) / max(setup_time, 1e-9)
            priority = (3 - j.priority) / 2.0
            risk = j.quality_risk
            score = (
                0.25 * proc
                + 0.30 * due_pressure
                + 0.20 * setup
                + 0.20 * priority
                + 0.05 * risk
            )
            return (float(score), j.job_id)

        return min(queue, key=composite_key)

    raise ValueError(f"unsupported dispatch rule: {rule}")  # pragma: no cover


def rule_from_action(action: int | np.integer) -> DispatchRule:
    try:
        return DispatchRule(int(action))
    except ValueError as exc:
        raise ValueError(f"action must be in [0, {len(DispatchRule) - 1}]") from exc
