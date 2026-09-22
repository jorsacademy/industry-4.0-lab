from __future__ import annotations

import numpy as np
import pytest

from dmdtrl.fjsp_baselines import run_fjsp_policy, shortest_processing_action
from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_models import (
    FJSPAction,
    FJSPInstance,
    FJSPJob,
    FJSPMachineOption,
    FJSPOperation,
)
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


def _operation(job_id: int, index: int, *options: tuple[int, float]) -> FJSPOperation:
    return FJSPOperation(
        job_id=job_id,
        operation_index=index,
        machine_options=tuple(
            FJSPMachineOption(machine_id=machine_id, processing_time=processing_time)
            for machine_id, processing_time in options
        ),
    )


def test_model_requires_consecutive_precedence_chain() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        FJSPJob(
            job_id=0,
            release_time=0.0,
            due_date=10.0,
            priority=1,
            family=0,
            operations=(_operation(0, 1, (0, 2.0)),),
        )


def test_operation_rejects_duplicate_machine_options() -> None:
    with pytest.raises(ValueError, match="unique"):
        _operation(0, 0, (0, 2.0), (0, 3.0))


def test_generator_is_reproducible_and_flexible() -> None:
    cfg = FJSPGeneratorConfig(
        n_jobs=6,
        n_machines=4,
        operations_min=2,
        operations_max=3,
        eligible_machines_min=2,
        eligible_machines_max=3,
    )
    first = generate_fjsp_instance(np.random.default_rng(123), cfg)
    second = generate_fjsp_instance(np.random.default_rng(123), cfg)
    assert first == second
    assert all(len(job.operations) >= 2 for job in first.jobs)
    assert all(
        2 <= len(operation.machine_options) <= 3
        for job in first.jobs
        for operation in job.operations
    )


def test_simulator_enforces_precedence_and_machine_eligibility() -> None:
    job = FJSPJob(
        job_id=0,
        release_time=0.0,
        due_date=20.0,
        priority=2,
        family=1,
        operations=(
            _operation(0, 0, (0, 3.0), (1, 5.0)),
            _operation(0, 1, (1, 2.0)),
        ),
    )
    sim = FlexibleJobShopSimulator(FJSPInstance(jobs=(job,), n_machines=2))
    assert sim.current_time == 0.0
    assert sim.eligible_actions() == (
        FJSPAction(0, 0, 0),
        FJSPAction(0, 0, 1),
    )
    with pytest.raises(ValueError, match="not precedence/resource feasible"):
        sim.step_assignment(0, 1, 1)

    terminated = sim.step_assignment(0, 0, 0)
    assert terminated is False
    assert sim.current_time == pytest.approx(3.0)
    assert sim.eligible_actions() == (FJSPAction(0, 1, 1),)
    assert sim.step_assignment(0, 1, 1) is True
    assert sim.current_time == pytest.approx(5.0)
    assert sim.metrics()["weighted_tardiness"] == 0.0


def test_dynamic_release_and_sequence_setup_progression() -> None:
    first = FJSPJob(
        job_id=0,
        release_time=2.0,
        due_date=10.0,
        priority=1,
        family=0,
        operations=(_operation(0, 0, (0, 2.0)),),
    )
    second = FJSPJob(
        job_id=1,
        release_time=3.0,
        due_date=20.0,
        priority=1,
        family=1,
        operations=(_operation(1, 0, (0, 1.0)),),
    )
    sim = FlexibleJobShopSimulator(
        FJSPInstance(jobs=(first, second), n_machines=1),
        default_setup_time=1.5,
    )
    assert sim.current_time == pytest.approx(2.0)
    sim.step_assignment(0, 0, 0)
    assert sim.current_time == pytest.approx(4.0)
    sim.step_assignment(1, 0, 0)
    records = sim.schedule_records()
    assert records[1]["setup_time"] == pytest.approx(1.5)
    assert records[1]["completion_time"] == pytest.approx(6.5)
    assert sim.metrics()["total_setup_time"] == pytest.approx(1.5)


def test_greedy_policy_completes_generated_true_fjsp() -> None:
    cfg = FJSPGeneratorConfig(
        n_jobs=8,
        n_machines=4,
        operations_min=2,
        operations_max=4,
        eligible_machines_min=1,
        eligible_machines_max=3,
    )
    instance = generate_fjsp_instance(np.random.default_rng(44), cfg)
    sim = FlexibleJobShopSimulator(instance, default_setup_time=1.0)
    metrics = run_fjsp_policy(sim, shortest_processing_action)
    assert metrics["scheduled_operations"] == float(instance.total_operations)
    assert metrics["completed_jobs"] == float(len(instance.jobs))
    assert metrics["makespan"] > 0.0
    assert 0.0 < metrics["utilization"] <= 1.0
