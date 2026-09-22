from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dmdtrl.fjsp_models import FJSPInstance, FJSPJob, FJSPMachineOption, FJSPOperation


@dataclass(slots=True, frozen=True)
class FJSPGeneratorConfig:
    n_jobs: int = 20
    n_machines: int = 5
    n_families: int = 4
    operations_min: int = 2
    operations_max: int = 5
    eligible_machines_min: int = 1
    eligible_machines_max: int = 3
    mean_interarrival: float = 1.0
    processing_min: float = 2.0
    processing_max: float = 10.0
    due_date_factor_min: float = 1.5
    due_date_factor_max: float = 2.5

    def validate(self) -> None:
        if self.n_jobs <= 0 or self.n_machines <= 0 or self.n_families <= 0:
            raise ValueError("job, machine, and family counts must be positive")
        if self.operations_min <= 0 or self.operations_max < self.operations_min:
            raise ValueError("invalid operation-count range")
        if self.eligible_machines_min <= 0:
            raise ValueError("eligible_machines_min must be positive")
        if self.eligible_machines_max < self.eligible_machines_min:
            raise ValueError("invalid eligible-machine range")
        if self.eligible_machines_max > self.n_machines:
            raise ValueError("eligible_machines_max cannot exceed n_machines")
        if self.mean_interarrival <= 0.0:
            raise ValueError("mean_interarrival must be positive")
        if self.processing_min <= 0.0 or self.processing_max < self.processing_min:
            raise ValueError("invalid processing-time range")
        if self.due_date_factor_min <= 0.0 or self.due_date_factor_max < self.due_date_factor_min:
            raise ValueError("invalid due-date-factor range")


def generate_fjsp_instance(
    rng: np.random.Generator,
    config: FJSPGeneratorConfig | None = None,
) -> FJSPInstance:
    cfg = config or FJSPGeneratorConfig()
    cfg.validate()

    releases = np.cumsum(rng.exponential(cfg.mean_interarrival, size=cfg.n_jobs))
    priorities = rng.choice(np.array([1, 2, 3]), size=cfg.n_jobs, p=np.array([0.60, 0.30, 0.10]))
    families = rng.integers(0, cfg.n_families, size=cfg.n_jobs)

    jobs: list[FJSPJob] = []
    for job_id in range(cfg.n_jobs):
        operation_count = int(rng.integers(cfg.operations_min, cfg.operations_max + 1))
        operations: list[FJSPOperation] = []
        nominal_route_time = 0.0

        for operation_index in range(operation_count):
            eligible_count = int(
                rng.integers(cfg.eligible_machines_min, cfg.eligible_machines_max + 1)
            )
            machine_ids = np.sort(
                rng.choice(cfg.n_machines, size=eligible_count, replace=False)
            )
            base_processing = float(rng.uniform(cfg.processing_min, cfg.processing_max))
            options = tuple(
                FJSPMachineOption(
                    machine_id=int(machine_id),
                    processing_time=float(base_processing * rng.uniform(0.80, 1.20)),
                )
                for machine_id in machine_ids
            )
            nominal_route_time += min(option.processing_time for option in options)
            operations.append(
                FJSPOperation(
                    job_id=job_id,
                    operation_index=operation_index,
                    machine_options=options,
                )
            )

        due_factor = float(
            rng.uniform(cfg.due_date_factor_min, cfg.due_date_factor_max)
        )
        release_time = float(releases[job_id])
        due_date = release_time + nominal_route_time * due_factor + cfg.mean_interarrival * 2.0
        jobs.append(
            FJSPJob(
                job_id=job_id,
                release_time=release_time,
                due_date=float(due_date),
                priority=int(priorities[job_id]),
                family=int(families[job_id]),
                operations=tuple(operations),
            )
        )

    return FJSPInstance(jobs=tuple(jobs), n_machines=cfg.n_machines)
