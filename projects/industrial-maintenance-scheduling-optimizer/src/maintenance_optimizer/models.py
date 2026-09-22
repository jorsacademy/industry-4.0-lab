from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping


class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class MaintenanceType(Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"


@dataclass(frozen=True, slots=True)
class Asset:
    id: str
    name: str
    location: str
    criticality: Priority
    failure_rate: float
    maintenance_cost: float
    downtime_cost_per_hour: float
    last_maintenance: datetime
    next_due: datetime
    maintenance_duration: float
    resource_requirements: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class MaintenanceTask:
    id: str
    asset_id: str
    task_type: MaintenanceType
    priority: Priority
    duration: float
    earliest_start: datetime
    latest_finish: datetime
    resource_requirements: Mapping[str, int]
    cost: float


@dataclass(frozen=True, slots=True)
class Resource:
    id: str
    name: str
    type: str
    capacity: int
    cost_per_hour: float
    availability_schedule: Mapping[datetime, bool] = field(default_factory=dict)
