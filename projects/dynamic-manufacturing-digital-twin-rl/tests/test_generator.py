import numpy as np

from dmdtrl.generator import generate_jobs


def test_job_generation_is_reproducible_and_valid():
    kwargs = dict(
        n_jobs=25,
        n_families=4,
        mean_interarrival=2.5,
        processing_range=(2.0, 8.0),
        due_date_factor_range=(2.0, 5.0),
    )
    a = generate_jobs(np.random.default_rng(7), **kwargs)
    b = generate_jobs(np.random.default_rng(7), **kwargs)

    assert a == b
    assert len(a) == 25
    assert len({j.job_id for j in a}) == 25
    assert all(j.arrival_time >= 0 for j in a)
    assert all(j.due_date > j.arrival_time for j in a)
    assert all(1 <= j.priority <= 3 for j in a)
    assert all(0.0 <= j.quality_risk <= 1.0 for j in a)
