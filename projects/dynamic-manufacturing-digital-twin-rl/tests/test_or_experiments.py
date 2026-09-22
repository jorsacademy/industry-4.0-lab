from __future__ import annotations

from dmdtrl.env import EnvConfig
from dmdtrl.or_experiments import evaluate_cpsat_policy, run_cpsat_policy
from dmdtrl.or_policy import CPSATDecision


class FirstAvailableScheduler:
    name = "FAKE_OR"

    def choose(self, env):
        job = env.queued_jobs()[0]
        machine = env.available_machines()[0]
        return CPSATDecision(
            job_id=job.job_id,
            machine_id=machine.machine_id,
            solver_status="FEASIBLE",
            objective_value=0.0,
            horizon_jobs=1,
        )


def test_run_cpsat_policy_uses_standard_metric_schema() -> None:
    row = run_cpsat_policy(
        FirstAvailableScheduler(),
        seed=5,
        config=EnvConfig(
            n_jobs=6,
            n_machines=2,
            mean_interarrival=0.2,
            breakdown_probability=0.0,
        ),
    )
    assert row["policy"] == "FAKE_OR"
    assert row["seed"] == 5
    assert row["completed_jobs"] == 6.0
    assert float(row["mean_decision_time_ms"]) >= 0.0


def test_evaluate_cpsat_policy_requires_seeds() -> None:
    try:
        evaluate_cpsat_policy(FirstAvailableScheduler(), [])
    except ValueError as exc:
        assert "evaluation seed" in str(exc)
    else:
        raise AssertionError("expected ValueError")
