from datetime import datetime, timedelta

from maintenance_optimizer import Asset, MaintenanceScheduler, Priority, Resource


def main() -> None:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    scheduler = MaintenanceScheduler(now=now)
    scheduler.add_asset(
        Asset(
            id="PUMP_001",
            name="Main Water Pump",
            location="Building A",
            criticality=Priority.CRITICAL,
            failure_rate=0.15,
            maintenance_cost=2500,
            downtime_cost_per_hour=1000,
            last_maintenance=now - timedelta(days=60),
            next_due=now + timedelta(days=5),
            maintenance_duration=8,
            resource_requirements={"technician": 2, "equipment": 1},
        )
    )
    scheduler.add_resource(Resource("TECH", "Maintenance Technicians", "technician", 8, 75))
    scheduler.add_resource(Resource("EQUIP", "Maintenance Equipment", "equipment", 4, 50))
    scheduler.generate_maintenance_tasks(planning_horizon_days=90)
    schedule = scheduler.greedy_schedule(planning_horizon_days=30)
    metrics = scheduler.calculate_schedule_metrics(schedule)
    print(scheduler.generate_schedule_report(schedule, metrics))


if __name__ == "__main__":
    main()
