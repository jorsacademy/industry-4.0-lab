from datetime import datetime, timedelta

import numpy as np

from maintenance_optimizer import (
    Asset,
    MaintenanceScheduler,
    MaintenanceTask,
    MaintenanceType,
    Priority,
    Resource,
)

NOW = datetime(2026, 1, 1, 8, 0)


def make_scheduler() -> MaintenanceScheduler:
    scheduler = MaintenanceScheduler(now=NOW)
    scheduler.add_asset(
        Asset(
            id="PUMP_001",
            name="Main Pump",
            location="Plant A",
            criticality=Priority.CRITICAL,
            failure_rate=0.2,
            maintenance_cost=1000,
            downtime_cost_per_hour=2000,
            last_maintenance=NOW - timedelta(days=30),
            next_due=NOW + timedelta(days=5),
            maintenance_duration=4,
            resource_requirements={"technician": 2},
        )
    )
    scheduler.add_resource(Resource("TECH", "Technicians", "technician", 2, 50))
    return scheduler


def test_preventive_generation_is_in_horizon():
    scheduler = make_scheduler()
    tasks = scheduler.generate_maintenance_tasks(30, rng=np.random.default_rng(123))
    preventive = [t for t in tasks if t.task_type is MaintenanceType.PREVENTIVE]
    assert len(preventive) == 1
    assert preventive[0].asset_id == "PUMP_001"


def test_priority_score_favors_emergency():
    scheduler = make_scheduler()
    common = dict(
        asset_id="PUMP_001",
        priority=Priority.CRITICAL,
        duration=1,
        earliest_start=NOW,
        latest_finish=NOW + timedelta(days=1),
        resource_requirements={"technician": 1},
        cost=100,
    )
    preventive = MaintenanceTask(id="p", task_type=MaintenanceType.PREVENTIVE, **common)
    emergency = MaintenanceTask(id="e", task_type=MaintenanceType.EMERGENCY, **common)
    assert scheduler.calculate_priority_score(emergency) > scheduler.calculate_priority_score(preventive)


def test_greedy_respects_resource_capacity_and_time_window():
    scheduler = make_scheduler()
    for index in range(2):
        scheduler.add_task(
            MaintenanceTask(
                id=f"T{index}",
                asset_id="PUMP_001",
                task_type=MaintenanceType.PREVENTIVE,
                priority=Priority.HIGH,
                duration=4,
                earliest_start=NOW,
                latest_finish=NOW + timedelta(hours=8),
                resource_requirements={"technician": 2},
                cost=100,
            )
        )
    schedule = scheduler.greedy_schedule(1)
    entries = sorted(schedule["PUMP_001"])
    assert len(entries) == 2
    assert entries[0][1] <= entries[1][0]
    assert entries[-1][1] <= NOW + timedelta(hours=8)


def test_unavailable_resource_blocks_task():
    scheduler = MaintenanceScheduler(now=NOW)
    scheduler.add_asset(make_scheduler().assets[0])
    unavailable = {NOW + timedelta(hours=i): False for i in range(4)}
    scheduler.add_resource(Resource("TECH", "Technicians", "technician", 2, 50, unavailable))
    scheduler.add_task(
        MaintenanceTask(
            id="T",
            asset_id="PUMP_001",
            task_type=MaintenanceType.PREVENTIVE,
            priority=Priority.HIGH,
            duration=4,
            earliest_start=NOW,
            latest_finish=NOW + timedelta(hours=4),
            resource_requirements={"technician": 1},
            cost=100,
        )
    )
    assert scheduler.greedy_schedule(1) == {}


def test_metrics_empty_schedule_has_zero_utilization():
    scheduler = make_scheduler()
    metrics = scheduler.calculate_schedule_metrics({})
    assert metrics["tasks_scheduled"] == 0
    assert metrics["average_resource_utilization"] == 0.0
