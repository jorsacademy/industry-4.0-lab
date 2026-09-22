from __future__ import annotations

import pytest

from dmdtrl.env import DynamicManufacturingEnv, EnvConfig
from dmdtrl.models import Job
from dmdtrl.or_policy import CPSATConfig, RollingHorizonCPSATPolicy, select_horizon


def test_cpsat_config_validates_positive_limits() -> None:
    with pytest.raises(ValueError):
        CPSATConfig(max_jobs=0)
    with pytest.raises(ValueError):
        CPSATConfig(time_limit_s=0.0)
    with pytest.raises(ValueError):
        CPSATConfig(time_scale=0)


def test_select_horizon_is_deterministic_and_urgency_biased() -> None:
    jobs = (
        Job(0, 0.0, 5.0, 20.0, 1, 0, 0.1),
        Job(1, 0.0, 5.0, 10.0, 1, 0, 0.1),
        Job(2, 0.0, 5.0, 10.0, 3, 0, 0.1),
    )
    selected = select_horizon(jobs, max_jobs=2)
    assert [job.job_id for job in selected] == [2, 1]


def test_explicit_assignment_uses_selected_job_and_machine() -> None:
    env = DynamicManufacturingEnv(
        EnvConfig(
            n_jobs=4,
            n_machines=2,
            mean_interarrival=0.01,
            breakdown_probability=0.0,
        )
    )
    env.reset(seed=7)
    job = env.queued_jobs()[0]
    machine = env.available_machines()[-1]

    env.step_assignment(job.job_id, machine.machine_id, decision_label="TEST")

    operation = env.schedule[-1]
    assert operation.job_id == job.job_id
    assert operation.machine_id == machine.machine_id
    assert operation.rule == "TEST"


def test_explicit_assignment_rejects_unknown_job_and_unavailable_machine() -> None:
    env = DynamicManufacturingEnv(
        EnvConfig(
            n_jobs=5,
            n_machines=2,
            mean_interarrival=0.01,
            breakdown_probability=0.0,
        )
    )
    env.reset(seed=11)
    job = env.queued_jobs()[0]

    with pytest.raises(ValueError, match="released queue"):
        env.step_assignment(999, 0)

    env.step_assignment(job.job_id, 0)
    assert env.queued_jobs()
    with pytest.raises(ValueError, match="currently available"):
        env.step_assignment(env.queued_jobs()[0].job_id, 0)


def test_cpsat_returns_released_job_and_available_machine() -> None:
    pytest.importorskip("ortools")
    env = DynamicManufacturingEnv(
        EnvConfig(
            n_jobs=8,
            n_machines=2,
            mean_interarrival=0.05,
            breakdown_probability=0.0,
        )
    )
    env.reset(seed=23)
    policy = RollingHorizonCPSATPolicy(CPSATConfig(max_jobs=6, time_limit_s=1.0))
    decision = policy.choose(env)

    assert decision.job_id in {job.job_id for job in env.queued_jobs()}
    assert decision.machine_id in {machine.machine_id for machine in env.available_machines()}
    assert decision.solver_status in {"OPTIMAL", "FEASIBLE"}
    assert not decision.used_fallback


def test_timeout_fallback_returns_deterministic_feasible_assignment() -> None:
    env = DynamicManufacturingEnv(
        EnvConfig(
            n_jobs=6,
            n_machines=2,
            mean_interarrival=0.01,
            breakdown_probability=0.0,
        )
    )
    env.reset(seed=31)
    policy = RollingHorizonCPSATPolicy(CPSATConfig(max_jobs=4, time_limit_s=0.01))
    jobs = select_horizon(env.queued_jobs(), max_jobs=4)
    machines = env.available_machines()

    first = policy._fallback_decision(env, jobs, machines, "UNKNOWN")
    second = policy._fallback_decision(env, jobs, machines, "UNKNOWN")

    assert first == second
    assert first.used_fallback
    assert first.solver_status == "UNKNOWN"
    assert first.job_id in {job.job_id for job in jobs}
    assert first.machine_id in {machine.machine_id for machine in machines}
