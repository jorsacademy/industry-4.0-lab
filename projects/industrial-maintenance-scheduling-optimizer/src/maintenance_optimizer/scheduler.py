from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .models import Asset, MaintenanceTask, MaintenanceType, Priority, Resource

Schedule = Dict[str, List[Tuple[datetime, datetime, str]]]


class MaintenanceScheduler:
    """Resource-constrained maintenance scheduler.

    The LP method is intentionally described as an LP relaxation. SciPy's
    ``linprog`` does not enforce binary start variables. For discrete schedules,
    use ``greedy_schedule`` or replace the relaxation with a MILP/CP-SAT solver.
    """

    def __init__(self, *, now: Optional[datetime] = None) -> None:
        self.assets: list[Asset] = []
        self.tasks: list[MaintenanceTask] = []
        self.resources: list[Resource] = []
        self.schedule: Schedule = {}
        self._now = now

    def now(self) -> datetime:
        return self._now or datetime.now()

    def add_asset(self, asset: Asset) -> None:
        if any(existing.id == asset.id for existing in self.assets):
            raise ValueError(f"duplicate asset id: {asset.id}")
        self.assets.append(asset)

    def add_resource(self, resource: Resource) -> None:
        if resource.capacity < 0:
            raise ValueError("resource capacity must be non-negative")
        if any(existing.id == resource.id for existing in self.resources):
            raise ValueError(f"duplicate resource id: {resource.id}")
        self.resources.append(resource)

    def add_task(self, task: MaintenanceTask) -> None:
        if task.duration <= 0:
            raise ValueError("task duration must be positive")
        if task.latest_finish <= task.earliest_start:
            raise ValueError("latest_finish must be after earliest_start")
        if not any(asset.id == task.asset_id for asset in self.assets):
            raise ValueError(f"unknown asset id: {task.asset_id}")
        self.tasks.append(task)

    def generate_maintenance_tasks(
        self,
        planning_horizon_days: int = 365,
        *,
        preventive_interval_days: int = 90,
        rng: Optional[np.random.Generator] = None,
    ) -> list[MaintenanceTask]:
        """Generate preventive and stochastic predictive tasks."""
        if planning_horizon_days <= 0:
            raise ValueError("planning_horizon_days must be positive")
        if preventive_interval_days <= 0:
            raise ValueError("preventive_interval_days must be positive")

        rng = rng or np.random.default_rng()
        current_date = self.now()
        end_date = current_date + timedelta(days=planning_horizon_days)
        generated: list[MaintenanceTask] = []

        for asset in self.assets:
            next_maintenance = asset.next_due
            counter = 1
            while next_maintenance <= end_date:
                generated.append(
                    MaintenanceTask(
                        id=f"PM_{asset.id}_{counter}",
                        asset_id=asset.id,
                        task_type=MaintenanceType.PREVENTIVE,
                        priority=asset.criticality,
                        duration=asset.maintenance_duration,
                        earliest_start=next_maintenance - timedelta(days=7),
                        latest_finish=next_maintenance + timedelta(days=3),
                        resource_requirements=asset.resource_requirements,
                        cost=asset.maintenance_cost,
                    )
                )
                next_maintenance += timedelta(days=preventive_interval_days)
                counter += 1

            if asset.failure_rate > 0.1:
                months = min(12, ceil(planning_horizon_days / 30))
                monthly_probability = min(max(asset.failure_rate / 12.0, 0.0), 1.0)
                for month in range(months):
                    if rng.random() < monthly_probability:
                        task_date = current_date + timedelta(days=30 * month)
                        if task_date <= end_date:
                            generated.append(
                                MaintenanceTask(
                                    id=f"PD_{asset.id}_{month}",
                                    asset_id=asset.id,
                                    task_type=MaintenanceType.PREDICTIVE,
                                    priority=Priority.HIGH,
                                    duration=asset.maintenance_duration * 0.7,
                                    earliest_start=task_date - timedelta(days=5),
                                    latest_finish=task_date + timedelta(days=5),
                                    resource_requirements=asset.resource_requirements,
                                    cost=asset.maintenance_cost * 0.8,
                                )
                            )

        self.tasks.extend(generated)
        return generated

    def calculate_priority_score(self, task: MaintenanceTask) -> float:
        asset = self._asset(task.asset_id)
        priority_weights = {
            Priority.CRITICAL: 100.0,
            Priority.HIGH: 75.0,
            Priority.MEDIUM: 50.0,
            Priority.LOW: 25.0,
        }
        type_weights = {
            MaintenanceType.EMERGENCY: 2.0,
            MaintenanceType.CORRECTIVE: 1.5,
            MaintenanceType.PREDICTIVE: 1.2,
            MaintenanceType.PREVENTIVE: 1.0,
        }
        days_until_deadline = (task.latest_finish - self.now()).total_seconds() / 86400
        urgency_factor = max(0.0, min(1.0, 1.0 - days_until_deadline / 30.0))
        criticality_factor = asset.downtime_cost_per_hour / 1000.0
        return (
            priority_weights[task.priority] + urgency_factor * 50.0 + criticality_factor
        ) * type_weights[task.task_type]

    def greedy_schedule(self, planning_horizon_days: int = 30) -> Schedule:
        horizon_start = self.now()
        horizon_end = horizon_start + timedelta(days=planning_horizon_days)
        relevant = [
            task
            for task in self.tasks
            if task.earliest_start < horizon_end and task.latest_finish > horizon_start
        ]
        resource_usage: dict[str, dict[datetime, int]] = {
            r.id: defaultdict(int) for r in self.resources
        }
        schedule: Schedule = {}

        for task in sorted(relevant, key=self.calculate_priority_score, reverse=True):
            earliest = max(task.earliest_start, horizon_start)
            feasible = self.find_feasible_start_time(task, earliest, resource_usage)
            if feasible is None:
                continue
            end = feasible + timedelta(hours=task.duration)
            if end > min(task.latest_finish, horizon_end):
                continue
            schedule.setdefault(task.asset_id, []).append((feasible, end, task.id))
            self.update_resource_usage(task, feasible, end, resource_usage)

        self.schedule = schedule
        return schedule

    def optimize_schedule_lp_relaxation(self, planning_horizon_days: int = 30) -> Schedule:
        """Solve a sparse LP relaxation and only return near-integral starts."""
        start = self.now()
        horizon_end = start + timedelta(days=planning_horizon_days)
        tasks = [
            t for t in self.tasks if t.earliest_start < horizon_end and t.latest_finish > start
        ]
        slots = planning_horizon_days * 24
        if not tasks:
            return {}

        n_vars = len(tasks) * slots
        c = np.full(n_vars, 1e6, dtype=float)
        bounds: list[tuple[float, float]] = [(0.0, 0.0)] * n_vars

        for i, task in enumerate(tasks):
            score = self.calculate_priority_score(task)
            for slot in range(slots):
                slot_start = start + timedelta(hours=slot)
                slot_end = slot_start + timedelta(hours=task.duration)
                idx = i * slots + slot
                if (
                    slot_start >= task.earliest_start
                    and slot_end <= task.latest_finish
                    and slot_end <= horizon_end
                ):
                    c[idx] = task.cost - score
                    bounds[idx] = (0.0, 1.0)

        a_eq = lil_matrix((len(tasks), n_vars), dtype=float)
        for i in range(len(tasks)):
            a_eq[i, i * slots : (i + 1) * slots] = 1.0
        b_eq = np.ones(len(tasks), dtype=float)

        rows = len(self.resources) * slots
        a_ub = lil_matrix((rows, n_vars), dtype=float)
        b_ub = np.zeros(rows, dtype=float)
        row = 0
        for resource in self.resources:
            for t_slot in range(slots):
                b_ub[row] = resource.capacity
                instant = start + timedelta(hours=t_slot)
                if (
                    resource.availability_schedule
                    and resource.availability_schedule.get(instant) is False
                ):
                    b_ub[row] = 0
                for i, task in enumerate(tasks):
                    qty = task.resource_requirements.get(resource.type)
                    if not qty:
                        continue
                    duration_slots = ceil(task.duration)
                    for s in range(max(0, t_slot - duration_slots + 1), t_slot + 1):
                        a_ub[row, i * slots + s] = qty
                row += 1

        result = linprog(
            c,
            A_ub=a_ub.tocsr(),
            b_ub=b_ub,
            A_eq=a_eq.tocsr(),
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return self.greedy_schedule(planning_horizon_days)

        x = result.x.reshape(len(tasks), slots)
        schedule: Schedule = {}
        for i, task in enumerate(tasks):
            chosen = np.flatnonzero(x[i] >= 1.0 - 1e-8)
            if len(chosen) != 1:
                continue
            slot = int(chosen[0])
            task_start = start + timedelta(hours=slot)
            task_end = task_start + timedelta(hours=task.duration)
            schedule.setdefault(task.asset_id, []).append((task_start, task_end, task.id))
        return schedule

    def find_feasible_start_time(
        self,
        task: MaintenanceTask,
        earliest_start: datetime,
        resource_usage: dict[str, dict[datetime, int]],
    ) -> Optional[datetime]:
        current = self._ceil_hour(earliest_start)
        latest_start = task.latest_finish - timedelta(hours=task.duration)
        while current <= latest_start:
            if self.check_resource_availability(task, current, resource_usage):
                return current
            current += timedelta(hours=1)
        return None

    def check_resource_availability(
        self,
        task: MaintenanceTask,
        start_time: datetime,
        resource_usage: dict[str, dict[datetime, int]],
    ) -> bool:
        end_time = start_time + timedelta(hours=task.duration)
        resources_by_type: dict[str, list[Resource]] = defaultdict(list)
        for resource in self.resources:
            resources_by_type[resource.type].append(resource)

        for resource_type, required in task.resource_requirements.items():
            candidates = resources_by_type.get(resource_type, [])
            if not candidates:
                return False
            current = start_time
            while current < end_time:
                available_capacity = 0
                for resource in candidates:
                    if (
                        resource.availability_schedule
                        and resource.availability_schedule.get(current) is False
                    ):
                        continue
                    available_capacity += resource.capacity - resource_usage[resource.id].get(
                        current, 0
                    )
                if available_capacity < required:
                    return False
                current += timedelta(hours=1)
        return True

    def update_resource_usage(
        self,
        task: MaintenanceTask,
        start_time: datetime,
        end_time: datetime,
        resource_usage: dict[str, dict[datetime, int]],
    ) -> None:
        resources_by_type: dict[str, list[Resource]] = defaultdict(list)
        for resource in self.resources:
            resources_by_type[resource.type].append(resource)

        for resource_type, required in task.resource_requirements.items():
            remaining = required
            for resource in resources_by_type.get(resource_type, []):
                current = start_time
                allocatable = resource.capacity
                while current < end_time:
                    if (
                        resource.availability_schedule
                        and resource.availability_schedule.get(current) is False
                    ):
                        allocatable = 0
                        break
                    allocatable = min(
                        allocatable,
                        resource.capacity - resource_usage[resource.id].get(current, 0),
                    )
                    current += timedelta(hours=1)
                take = min(remaining, max(allocatable, 0))
                if take:
                    current = start_time
                    while current < end_time:
                        resource_usage[resource.id][current] += take
                        current += timedelta(hours=1)
                    remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise RuntimeError("resource usage update called for infeasible task")

    def calculate_schedule_metrics(self, schedule: Optional[Schedule] = None) -> dict:
        schedule = self.schedule if schedule is None else schedule
        total_cost = 0.0
        total_downtime = 0.0
        tasks_scheduled = 0
        tasks_overdue = 0
        resource_used_hours = defaultdict(float)

        for _, entries in schedule.items():
            for start_time, _, task_id in entries:
                task = self._task(task_id)
                total_cost += task.cost
                total_downtime += task.duration
                tasks_scheduled += 1
                if start_time + timedelta(hours=task.duration) > task.latest_finish:
                    tasks_overdue += 1
                for resource in self.resources:
                    qty = task.resource_requirements.get(resource.type, 0)
                    resource_used_hours[resource.id] += task.duration * qty

        utilization = {}
        for resource in self.resources:
            capacity_hours = resource.capacity * 24 * 30
            utilization[resource.id] = (
                100.0 * resource_used_hours[resource.id] / capacity_hours
                if capacity_hours
                else 0.0
            )

        avg = float(np.mean(list(utilization.values()))) if utilization else 0.0
        return {
            "total_cost": total_cost,
            "total_downtime_hours": total_downtime,
            "tasks_scheduled": tasks_scheduled,
            "tasks_overdue": tasks_overdue,
            "schedule_efficiency": 100.0
            * (tasks_scheduled - tasks_overdue)
            / max(tasks_scheduled, 1),
            "resource_utilization": utilization,
            "average_resource_utilization": avg,
        }

    def generate_schedule_report(self, schedule: Schedule, metrics: dict) -> str:
        lines = [
            "=== MAINTENANCE SCHEDULE OPTIMIZATION REPORT ===",
            "",
            "SCHEDULE METRICS:",
            f"Total Cost: ${metrics['total_cost']:,.2f}",
            f"Total Planned Downtime: {metrics['total_downtime_hours']:.1f} hours",
            f"Tasks Scheduled: {metrics['tasks_scheduled']}",
            f"Tasks Overdue: {metrics['tasks_overdue']}",
            f"Schedule Efficiency: {metrics['schedule_efficiency']:.1f}%",
            f"Average Resource Utilization: {metrics['average_resource_utilization']:.1f}%",
            "",
            "DETAILED SCHEDULE:",
        ]
        for asset_id, entries in sorted(schedule.items()):
            lines.append(f"Asset: {self._asset(asset_id).name} ({asset_id})")
            for start_time, end_time, task_id in sorted(entries):
                task = self._task(task_id)
                lines.append(
                    f"  {start_time:%Y-%m-%d %H:%M} - {end_time:%Y-%m-%d %H:%M}: "
                    f"{task.task_type.value.title()} Maintenance (${task.cost:.2f})"
                )
        return "\n".join(lines)

    def _asset(self, asset_id: str) -> Asset:
        try:
            return next(asset for asset in self.assets if asset.id == asset_id)
        except StopIteration as exc:
            raise KeyError(f"unknown asset id: {asset_id}") from exc

    def _task(self, task_id: str) -> MaintenanceTask:
        try:
            return next(task for task in self.tasks if task.id == task_id)
        except StopIteration as exc:
            raise KeyError(f"unknown task id: {task_id}") from exc

    @staticmethod
    def _ceil_hour(value: datetime) -> datetime:
        if value.minute == value.second == value.microsecond == 0:
            return value
        return value.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
