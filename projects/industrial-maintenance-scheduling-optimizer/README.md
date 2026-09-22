# Industrial Maintenance Scheduling Optimizer

A Python toolkit for resource-constrained preventive and predictive maintenance scheduling. It models assets, maintenance tasks, capacities, time windows, priority scoring, schedule metrics, and reporting.

> **License:** source-available for non-commercial use only. Commercial use is prohibited unless separately licensed in writing. See `LICENSE` and `NOTICE`.

## What it provides

- Asset, maintenance-task, and resource domain models.
- Preventive task generation and stochastic predictive task generation.
- Priority scoring using criticality, deadline urgency, downtime cost, and maintenance type.
- Deterministic greedy scheduling with resource-capacity and time-window checks.
- Resource availability calendars.
- Sparse SciPy LP relaxation for experimentation.
- Schedule metrics and text reports.
- Pytest test suite and GitHub Actions CI across Python 3.10-3.13.

## Important optimization note

The original prototype represented start decisions with `scipy.optimize.linprog` and `(0, 1)` bounds. Those bounds do **not** make variables binary. This repository therefore calls that method an **LP relaxation** and does not claim it is an exact MILP scheduler.

For exact discrete optimization at larger industrial scales, use a true MILP/CP-SAT formulation, decomposition, rolling horizons, or another scalable scheduling architecture. No claim is made that the LP relaxation can schedule 200,000 assets directly.

## Installation

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests examples
```

## Example

```python
from datetime import datetime, timedelta

from maintenance_optimizer import Asset, MaintenanceScheduler, Priority, Resource

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
```

A runnable version is available at `examples/basic_example.py`.

## Current scope

Preventive and predictive tasks can be generated automatically. Corrective and emergency maintenance types are represented in the domain model and priority scoring, but ingestion/generation of real failure events is intentionally left to the integrating application.

## Scaling

The greedy scheduler avoids the dense `tasks × hourly-slots` matrices used by the original prototype, but large industrial deployments still require workload partitioning, indexed data structures, rolling horizons, profiling, and a production-grade optimization strategy. Benchmark claims should be supported by reproducible benchmark data before publication.

## License

This project uses the **Jorsacademy Non-Commercial Source License 1.0**. Personal, non-commercial educational, and non-commercial research uses are permitted under its terms. Commercial use is prohibited without a separate written license.
