from __future__ import annotations

import numpy as np
import pytest

from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_models import FJSPAction, FJSPInstance, FJSPJob, FJSPMachineOption, FJSPOperation
from dmdtrl.fjsp_optimization import FJSPCPSATConfig, FJSPRollingHorizonCPSAT
from dmdtrl.fjsp_or_benchmark import FJSPBenchmarkConfig, run_benchmark, summarize_benchmark
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


def _small_generator() -> FJSPGeneratorConfig:
    return FJSPGeneratorConfig(
        n_jobs=4,
        n_machines=3,
        n_families=3,
        operations_min=2,
        operations_max=3,
        eligible_machines_min=1,
        eligible_machines_max=2,
        mean_interarrival=1.0,
        processing_min=1.0,
        processing_max=4.0,
    )


def test_fjsp_cpsat_config_validation() -> None:
    FJSPCPSATConfig().validate()
    with pytest.raises(ValueError, match="job_horizon"):
        FJSPCPSATConfig(job_horizon=0).validate()
    with pytest.raises(ValueError, match="solver_seconds"):
        FJSPCPSATConfig(solver_seconds=0.0).validate()
    with pytest.raises(ValueError, match="time_scale"):
        FJSPCPSATConfig(time_scale=0).validate()
    with pytest.raises(ValueError, match="random_seed"):
        FJSPCPSATConfig(random_seed=-1).validate()
    with pytest.raises(ValueError, match="num_search_workers"):
        FJSPCPSATConfig(num_search_workers=2).validate()


def test_candidate_horizon_never_uses_unreleased_jobs() -> None:
    instance = generate_fjsp_instance(np.random.default_rng(40001), _small_generator())
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=0.5)
    controller = FJSPRollingHorizonCPSAT(FJSPCPSATConfig(job_horizon=2, solver_seconds=0.2))

    candidates = controller._candidate_jobs(simulator)
    assert 1 <= len(candidates) <= 2
    assert all(job.release_time <= simulator.current_time + 1e-12 for job in candidates)
    assert all(simulator.next_operation[job.job_id] < len(job.operations) for job in candidates)


def test_candidate_horizon_keeps_currently_executable_job() -> None:
    job0 = FJSPJob(
        job_id=0,
        release_time=0.0,
        due_date=5.0,
        priority=3,
        family=0,
        operations=(
            FJSPOperation(0, 0, (FJSPMachineOption(0, 10.0),)),
            FJSPOperation(0, 1, (FJSPMachineOption(0, 1.0),)),
        ),
    )
    job1 = FJSPJob(
        job_id=1,
        release_time=0.0,
        due_date=100.0,
        priority=1,
        family=1,
        operations=(FJSPOperation(1, 0, (FJSPMachineOption(1, 1.0),)),),
    )
    simulator = FlexibleJobShopSimulator(FJSPInstance((job0, job1), n_machines=2))
    simulator.step(FJSPAction(0, 0, 0))
    assert simulator.current_time == pytest.approx(0.0)
    eligible = simulator.eligible_actions()
    assert eligible == (FJSPAction(1, 0, 1),)

    controller = FJSPRollingHorizonCPSAT(FJSPCPSATConfig(job_horizon=1, solver_seconds=0.2))
    candidates = controller._candidate_jobs(simulator, eligible)
    assert tuple(job.job_id for job in candidates) == (1,)
    decision = controller.choose(simulator)
    assert decision.action == FJSPAction(1, 0, 1)


def test_cpsat_returns_feasible_actions_and_completes_instance() -> None:
    instance = generate_fjsp_instance(np.random.default_rng(40002), _small_generator())
    simulator = FlexibleJobShopSimulator(instance, default_setup_time=0.5)
    controller = FJSPRollingHorizonCPSAT(
        FJSPCPSATConfig(job_horizon=4, solver_seconds=0.25, random_seed=7)
    )

    while not simulator.terminated:
        eligible = set(simulator.eligible_actions())
        decision = controller.choose(simulator)
        assert decision.action in eligible
        assert all(
            simulator.job(job_id).release_time <= simulator.current_time + 1e-12
            for job_id in decision.candidate_job_ids
        )
        assert decision.candidate_operations > 0
        if not decision.fallback:
            assert decision.objective_value is not None
            assert decision.solver_status in {"OPTIMAL", "FEASIBLE"}
        simulator.step(decision.action)

    metrics = simulator.metrics()
    stats = controller.stats()
    assert metrics["scheduled_operations"] == instance.total_operations
    assert metrics["completed_jobs"] == len(instance.jobs)
    assert stats["decision_count"] == instance.total_operations
    assert 0.0 <= stats["fallback_rate"] <= 1.0
    assert 0.0 <= stats["solver_success_rate"] <= 1.0


def test_fjsp_benchmark_runs_common_seed_controller_panel() -> None:
    rows = run_benchmark(
        FJSPBenchmarkConfig(seeds=1, seed_start=40003, default_setup_time=0.5),
        generator_config=FJSPGeneratorConfig(
            n_jobs=3,
            n_machines=3,
            n_families=2,
            operations_min=2,
            operations_max=2,
            eligible_machines_min=1,
            eligible_machines_max=2,
            processing_min=1.0,
            processing_max=3.0,
        ),
        cpsat_config=FJSPCPSATConfig(job_horizon=3, solver_seconds=0.25),
    )
    assert len(rows) == 3
    assert {row["policy"] for row in rows} == {
        "SHORTEST_PROCESSING",
        "EARLIEST_DUE_DATE",
        "ROLLING_HORIZON_CPSAT",
    }
    assert {row["seed"] for row in rows} == {40003}

    summary = summarize_benchmark(rows)
    assert len(summary) == 3
    assert float(summary[0]["weighted_tardiness"]) <= float(summary[-1]["weighted_tardiness"])


def test_phase5_benchmark_seed_boundary() -> None:
    with pytest.raises(ValueError, match="40000"):
        FJSPBenchmarkConfig(seed_start=39999).validate()
