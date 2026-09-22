from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

from dmdtrl.models import Job

if TYPE_CHECKING:
    from dmdtrl.env import DynamicManufacturingEnv
    from dmdtrl.models import Machine


@dataclass(frozen=True, slots=True)
class CPSATConfig:
    max_jobs: int = 12
    time_limit_s: float = 0.10
    time_scale: int = 100
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive")


@dataclass(frozen=True, slots=True)
class CPSATDecision:
    job_id: int
    machine_id: int
    solver_status: str
    objective_value: float | None
    horizon_jobs: int
    used_fallback: bool = False


def select_horizon(
    jobs: tuple[Job, ...],
    *,
    max_jobs: int,
) -> tuple[Job, ...]:
    """Select a deterministic urgency-biased rolling horizon from released jobs."""
    if max_jobs <= 0:
        raise ValueError("max_jobs must be positive")
    ordered = sorted(
        jobs,
        key=lambda job: (job.due_date, -job.priority, job.arrival_time, job.job_id),
    )
    return tuple(ordered[:max_jobs])


class RollingHorizonCPSATPolicy:
    """Receding-horizon CP-SAT scheduler over released jobs and available machines.

    The optimizer does not inspect future unreleased arrivals or future breakdown
    realizations. At every decision epoch it plans a bounded subset of the current
    queue, executes only the first job-machine assignment, then replans.

    If the declared online time budget expires before CP-SAT finds its first feasible
    solution, the controller executes a deterministic one-step emergency assignment.
    This preserves operational feasibility while exposing the event through
    ``used_fallback`` so compute-budget sensitivity can measure solver reliability.
    """

    name = "CP_SAT_RH"

    def __init__(self, config: CPSATConfig | None = None):
        self.config = config or CPSATConfig()

    def _fallback_decision(
        self,
        env: DynamicManufacturingEnv,
        jobs: tuple[Job, ...],
        machines: tuple[Machine, ...],
        solver_status: str,
    ) -> CPSATDecision:
        """Return a deterministic feasible assignment after an online solver timeout."""
        candidates: list[tuple[float, float, float, float, float, int, int, int]] = []
        for machine in machines:
            for job in jobs:
                setup = (
                    0.0
                    if machine.last_family in (None, job.family)
                    else env.config.sequence_setup_time
                )
                processing = job.processing_time / machine.speed
                completion = env.current_time + setup + processing
                tardiness = max(completion - job.due_date, 0.0)
                candidates.append(
                    (
                        job.priority * tardiness,
                        tardiness,
                        setup,
                        completion,
                        job.due_date,
                        -job.priority,
                        machine.machine_id,
                        job.job_id,
                    )
                )

        if not candidates:  # pragma: no cover - guarded by choose
            raise RuntimeError("fallback requires at least one job-machine candidate")

        *_, machine_id, job_id = min(candidates)
        return CPSATDecision(
            job_id=job_id,
            machine_id=machine_id,
            solver_status=solver_status,
            objective_value=None,
            horizon_jobs=len(jobs),
            used_fallback=True,
        )

    def choose(self, env: DynamicManufacturingEnv) -> CPSATDecision:
        jobs = select_horizon(env.queued_jobs(), max_jobs=self.config.max_jobs)
        machines = env.available_machines()
        if not jobs:
            raise RuntimeError("CP-SAT requires at least one released job")
        if not machines:
            raise RuntimeError("CP-SAT requires at least one available machine")

        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install OR dependencies with: pip install -e ".[or]"') from exc

        scale = self.config.time_scale
        now = int(round(env.current_time * scale))
        setup_duration = int(round(env.config.sequence_setup_time * scale))

        processing: dict[tuple[int, int], int] = {}
        max_single = 0
        for job in jobs:
            for machine in machines:
                duration = max(1, int(ceil((job.processing_time / machine.speed) * scale)))
                processing[(job.job_id, machine.machine_id)] = duration
                max_single = max(max_single, duration + setup_duration)

        due_dates = {job.job_id: int(round(job.due_date * scale)) for job in jobs}
        planning_span = sum(
            max(processing[(job.job_id, machine.machine_id)] for machine in machines)
            + setup_duration
            for job in jobs
        )
        upper = max(now + planning_span + max_single, max(due_dates.values()) + planning_span)

        model = cp_model.CpModel()
        start = {
            job.job_id: model.NewIntVar(now, upper, f"start_{job.job_id}") for job in jobs
        }
        end = {job.job_id: model.NewIntVar(now, upper, f"end_{job.job_id}") for job in jobs}
        tardiness = {
            job.job_id: model.NewIntVar(0, upper, f"tard_{job.job_id}") for job in jobs
        }

        assigned: dict[tuple[int, int], object] = {}
        for job in jobs:
            literals = []
            for machine in machines:
                key = (job.job_id, machine.machine_id)
                literal = model.NewBoolVar(f"assign_{job.job_id}_{machine.machine_id}")
                assigned[key] = literal
                literals.append(literal)
                model.Add(
                    end[job.job_id]
                    == start[job.job_id] + processing[(job.job_id, machine.machine_id)]
                ).OnlyEnforceIf(literal)
            model.Add(sum(literals) == 1)
            model.Add(tardiness[job.job_id] >= end[job.job_id] - due_dates[job.job_id])

        setup_terms = []
        first_arc: dict[tuple[int, int], object] = {}
        for machine in machines:
            machine_id = machine.machine_id
            empty = model.NewBoolVar(f"empty_{machine_id}")
            arcs = [(0, 0, empty)]
            machine_assignments = []

            for node_index, job in enumerate(jobs, start=1):
                assignment = assigned[(job.job_id, machine_id)]
                machine_assignments.append(assignment)

                self_loop = model.NewBoolVar(f"self_{machine_id}_{job.job_id}")
                model.Add(self_loop + assignment == 1)
                arcs.append((node_index, node_index, self_loop))

                first = model.NewBoolVar(f"first_{machine_id}_{job.job_id}")
                last = model.NewBoolVar(f"last_{machine_id}_{job.job_id}")
                first_arc[(job.job_id, machine_id)] = first
                arcs.append((0, node_index, first))
                arcs.append((node_index, 0, last))

                initial_setup = 0 if machine.last_family in (None, job.family) else setup_duration
                model.Add(start[job.job_id] >= now + initial_setup).OnlyEnforceIf(first)
                if initial_setup:
                    setup_terms.append(initial_setup * first)

            model.Add(sum(machine_assignments) == 0).OnlyEnforceIf(empty)
            model.Add(sum(machine_assignments) >= 1).OnlyEnforceIf(empty.Not())

            for from_index, from_job in enumerate(jobs, start=1):
                for to_index, to_job in enumerate(jobs, start=1):
                    if from_job.job_id == to_job.job_id:
                        continue
                    arc = model.NewBoolVar(
                        f"arc_{machine_id}_{from_job.job_id}_{to_job.job_id}"
                    )
                    arcs.append((from_index, to_index, arc))
                    transition_setup = 0 if from_job.family == to_job.family else setup_duration
                    model.Add(
                        start[to_job.job_id] >= end[from_job.job_id] + transition_setup
                    ).OnlyEnforceIf(arc)
                    if transition_setup:
                        setup_terms.append(transition_setup * arc)

            model.AddCircuit(arcs)

        makespan = model.NewIntVar(now, upper, "makespan")
        model.AddMaxEquality(makespan, [end[job.job_id] for job in jobs])

        weighted_tardiness = sum(job.priority * tardiness[job.job_id] for job in jobs)
        setup_cost = sum(setup_terms) if setup_terms else 0
        max_secondary = 10 * setup_duration * len(jobs) + upper
        primary_weight = max_secondary + 1
        model.Minimize(primary_weight * weighted_tardiness + 10 * setup_cost + makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit_s
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self.config.random_seed

        status = solver.Solve(model)
        if status == cp_model.UNKNOWN:
            return self._fallback_decision(
                env,
                jobs,
                machines,
                solver.StatusName(status),
            )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP-SAT did not produce a feasible plan: {solver.StatusName(status)}")

        first_choices: list[tuple[int, int, int]] = []
        for job in jobs:
            for machine in machines:
                key = (job.job_id, machine.machine_id)
                if solver.Value(first_arc[key]):
                    first_choices.append(
                        (solver.Value(start[job.job_id]), machine.machine_id, job.job_id)
                    )

        if not first_choices:
            raise RuntimeError("CP-SAT returned a plan without an executable first decision")

        _, machine_id, job_id = min(first_choices)
        return CPSATDecision(
            job_id=job_id,
            machine_id=machine_id,
            solver_status=solver.StatusName(status),
            objective_value=float(solver.ObjectiveValue()),
            horizon_jobs=len(jobs),
        )
