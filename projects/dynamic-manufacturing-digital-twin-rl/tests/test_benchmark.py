from dmdtrl.env import EnvConfig
from dmdtrl.evaluate import benchmark


def test_benchmark_returns_all_dispatching_rules():
    rows = benchmark(
        seeds=[0, 1],
        config=EnvConfig(n_jobs=12, n_machines=2, breakdown_probability=0.0),
    )
    assert len(rows) == 8
    assert len({row["policy"] for row in rows}) == 8
    assert all(row["makespan"] > 0 for row in rows)
