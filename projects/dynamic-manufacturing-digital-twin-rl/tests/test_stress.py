from dmdtrl.env import EnvConfig
from dmdtrl.or_policy import CPSATDecision
from dmdtrl.research import fixed_policies
from dmdtrl.scenarios import select_scenarios
from dmdtrl.stress import compare_candidate_across_scenarios, evaluate_scenarios


class FirstAvailableORScheduler:
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


def test_evaluate_scenarios_retains_scenario_and_seed_level_results():
    policies = fixed_policies()[:2]
    scenarios = select_scenarios(["nominal", "demand_120"])
    config = EnvConfig(n_jobs=8, n_machines=2)

    raw_rows, summary_rows = evaluate_scenarios(
        policies,
        [101, 102],
        scenarios,
        config,
        n_bootstrap=100,
    )

    assert len(raw_rows) == 8
    assert len(summary_rows) == 4
    assert {row["scenario"] for row in raw_rows} == {"nominal", "demand_120"}
    assert {row["seed"] for row in raw_rows} == {101, 102}
    assert all("weighted_tardiness_mean" in row for row in summary_rows)


def test_evaluate_scenarios_can_add_external_or_scheduler():
    policies = fixed_policies()[:1]
    scenarios = select_scenarios(["nominal", "demand_120"])
    config = EnvConfig(
        n_jobs=6,
        n_machines=2,
        mean_interarrival=0.2,
        breakdown_probability=0.0,
    )

    raw_rows, summary_rows = evaluate_scenarios(
        policies,
        [901],
        scenarios,
        config,
        cpsat_policy=FirstAvailableORScheduler(),
        n_bootstrap=100,
    )

    assert len(raw_rows) == 4
    assert len(summary_rows) == 4
    assert {row["policy"] for row in raw_rows} == {policies[0].name, "FAKE_OR"}
    assert {row["scenario"] for row in raw_rows} == {"nominal", "demand_120"}


def test_evaluate_scenarios_requires_seeds():
    try:
        evaluate_scenarios(fixed_policies()[:1], [], select_scenarios(["nominal"]))
    except ValueError as exc:
        assert "evaluation seed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_compare_candidate_across_scenarios_keeps_pairing_separate():
    rows = []
    for scenario, penalty in [("nominal", 0.0), ("stress", 10.0)]:
        for seed in range(4):
            rows.append(
                {
                    "scenario": scenario,
                    "policy": "candidate",
                    "seed": seed,
                    "weighted_tardiness": 5.0 + penalty,
                }
            )
            rows.append(
                {
                    "scenario": scenario,
                    "policy": "baseline",
                    "seed": seed,
                    "weighted_tardiness": 8.0 + penalty,
                }
            )

    comparisons = compare_candidate_across_scenarios(
        rows,
        candidate="candidate",
        baselines=["baseline"],
        metrics=("weighted_tardiness",),
        n_bootstrap=100,
        n_permutations=200,
    )

    assert len(comparisons) == 2
    assert {row["scenario"] for row in comparisons} == {"nominal", "stress"}
    assert all(row["mean_improvement"] == 3.0 for row in comparisons)
