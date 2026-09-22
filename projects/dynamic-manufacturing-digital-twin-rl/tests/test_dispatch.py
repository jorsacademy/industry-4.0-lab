from dmdtrl.dispatch import DispatchRule, select_job
from dmdtrl.models import Job, Machine


def _jobs():
    return [
        Job(0, 0.0, 6.0, 20.0, 1, 0, 0.1),
        Job(1, 1.0, 2.0, 12.0, 3, 1, 0.2),
        Job(2, 2.0, 4.0, 10.0, 2, 0, 0.3),
    ]


def test_core_dispatching_rules_choose_expected_jobs():
    jobs = _jobs()
    machine = Machine(machine_id=0, speed=1.0, last_family=0)

    assert select_job(jobs, machine, 5.0, DispatchRule.FIFO, 1.5).job_id == 0
    assert select_job(jobs, machine, 5.0, DispatchRule.HIGHEST_PRIORITY, 1.5).job_id == 1
    assert select_job(jobs, machine, 5.0, DispatchRule.EARLIEST_DUE_DATE, 1.5).job_id == 2
    assert select_job(jobs, machine, 5.0, DispatchRule.SHORTEST_PROCESSING_TIME, 1.5).job_id == 1
    assert select_job(jobs, machine, 5.0, DispatchRule.SAME_FAMILY_FIRST, 1.5).job_id in {0, 2}
    assert select_job(jobs, machine, 5.0, DispatchRule.MINIMUM_SETUP, 1.5).job_id in {0, 2}
