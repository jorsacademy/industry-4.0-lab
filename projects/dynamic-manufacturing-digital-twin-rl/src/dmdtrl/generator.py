from __future__ import annotations

import numpy as np

from dmdtrl.models import Job, Machine


def generate_jobs(
    rng: np.random.Generator,
    n_jobs: int,
    n_families: int,
    mean_interarrival: float,
    processing_range: tuple[float, float],
    due_date_factor_range: tuple[float, float],
) -> list[Job]:
    """Generate a reproducible stochastic order stream."""
    arrivals = np.cumsum(rng.exponential(mean_interarrival, size=n_jobs))
    processing = rng.uniform(processing_range[0], processing_range[1], size=n_jobs)
    due_factors = rng.uniform(due_date_factor_range[0], due_date_factor_range[1], size=n_jobs)
    priorities = rng.choice(np.array([1, 2, 3]), size=n_jobs, p=np.array([0.60, 0.30, 0.10]))
    families = rng.integers(0, n_families, size=n_jobs)
    quality_risks = rng.beta(2.0, 8.0, size=n_jobs)

    jobs: list[Job] = []
    for idx in range(n_jobs):
        due_date = float(
            arrivals[idx] + processing[idx] * due_factors[idx] + mean_interarrival * 2
        )
        jobs.append(
            Job(
                job_id=idx,
                arrival_time=float(arrivals[idx]),
                processing_time=float(processing[idx]),
                due_date=due_date,
                priority=int(priorities[idx]),
                family=int(families[idx]),
                quality_risk=float(quality_risks[idx]),
            )
        )
    return jobs


def generate_machines(
    rng: np.random.Generator,
    n_machines: int,
    speed_range: tuple[float, float],
) -> list[Machine]:
    """Generate heterogeneous parallel machines."""
    speeds = rng.uniform(speed_range[0], speed_range[1], size=n_machines)
    return [Machine(machine_id=i, speed=float(speeds[i])) for i in range(n_machines)]
