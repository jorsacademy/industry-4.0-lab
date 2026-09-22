from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from dmdtrl.fjsp_baselines import earliest_due_date_action
from dmdtrl.fjsp_models import FJSPAction
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


@dataclass(slots=True, frozen=True)
class FJSPCPSATConfig:
    job_horizon: int = 8
    solver_seconds: float = 0.10
    time_scale: int = 100
    random_seed: int = 0
    num_search_workers: int = 1

    def validate(self) -> None:
        if self.job_horizon <= 0:
            raise ValueError("job_horizon must be positive")
        if self.solver_seconds <= 0.0:
            raise ValueError("solver_seconds must be positive")
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.num_search_workers != 1:
            raise ValueError("num_search_workers must remain 1 for reproducible experiments")


@dataclass(slots=True, frozen=True)
class FJSPCPSATDecision:
    action: FJSPAction
    solver_status: str
    objective_value: float | None
    solve_time_seconds: float
    fallback: bool
    candidate_job_ids: tuple[int, ...]
    candidate_operations: int


class FJSPRollingHorizonCPSAT:
    """Released-job-only rolling-horizon CP-SAT controller for the FJSP core."""

    def __init__(self, config: FJSPCPSATConfig | None = None) -> None:
        self.config = config or FJSPCPSATConfig()
        self.config.validate()
        self.decision_count = 0
        self.fallback_count = 0
        self.total_solve_time_seconds = 0.0

    def choose(self, simulator: FlexibleJobShopSimulator) -> FJSPCPSATDecision:
        if simulator.terminated:
            raise RuntimeError("cannot optimize a completed FJSP schedule")
        eligible_now = simulator.eligible_actions()
        if not eligible_now:
            raise RuntimeError("simulator has no eligible action at the decision epoch")

        candidate_jobs = self._candidate_jobs(simulator, eligible_now)
        candidate_ids = tuple(job.job_id for job in candidate_jobs)
        candidate_id_set = set(candidate_ids)
        eligible_now = tuple(
            action for action in eligible_now if action.job_id in candidate_id_set
        )
        if not eligible_now:  # pragma: no cover - guaranteed by _candidate_jobs
            raise RuntimeError("candidate horizon excluded every currently feasible action")

        decision = self._solve(simulator, candidate_jobs, eligible_now)
        self.decision_count += 1
        self.total_solve_time_seconds += decision.solve_time_seconds
        if decision.fallback:
            self.fallback_count += 1
        return decision

    def stats(self) -> dict[str, float]:
        fallback_rate = self.fallback_count / max(self.decision_count, 1)
        return {
            "decision_count": float(self.decision_count),
            "fallback_count": float(self.fallback_count),
            "fallback_rate": fallback_rate,
            "mean_solve_time_ms": 1000.0
            * self.total_solve_time_seconds
            / max(self.decision_count, 1),
            "solver_success_rate": 1.0 - fallback_rate,
        }

    def _candidate_jobs(
        self,
        simulator: FlexibleJobShopSimulator,
        eligible_now: tuple[FJSPAction, ...] | list[FJSPAction] = (),
    ):
        feasible_job_ids = {action.job_id for action in eligible_now}
        released = [
            job
            for job in simulator.instance.jobs
            if job.release_time <= simulator.current_time + 1e-12
            and simulator.next_operation[job.job_id] < len(job.operations)
        ]
        released.sort(
            key=lambda job: (
                job.job_id not in feasible_job_ids,
                (job.due_date - simulator.current_time) / max(job.priority, 1),
                job.due_date,
                -job.priority,
                job.job_id,
            )
        )
        return tuple(released[: self.config.job_horizon])

    def _solve(self, simulator, candidate_jobs, eligible_now):
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:  # pragma: no cover - integration environment owns dependency
            raise RuntimeError(
                "OR-Tools is required for the FJSP CP-SAT controller; install the 'or' extra"
            ) from exc

        scale = self.config.time_scale
        current = _scale_ceil(simulator.current_time, scale)
        machine_available = {
            machine_id: _scale_ceil(machine.available_at, scale)
            for machine_id, machine in simulator.machines.items()
        }
        operation_keys: list[tuple[int, int]] = []
        operation_by_key: dict[tuple[int, int], Any] = {}
        job_by_id = {job.job_id: job for job in candidate_jobs}
        for job in candidate_jobs:
            first = simulator.next_operation[job.job_id]
            for operation in job.operations[first:]:
                key = (job.job_id, operation.operation_index)
                operation_keys.append(key)
                operation_by_key[key] = operation

        max_setup = max(
            [simulator.default_setup_time, *simulator.setup_times.values()],
            default=0.0,
        )
        max_base = max(
            [simulator.current_time, *[m.available_at for m in simulator.machines.values()]]
            + [job.due_date for job in candidate_jobs]
        )
        remaining_work = sum(
            max(option.processing_time for option in operation_by_key[key].machine_options)
            + max_setup
            for key in operation_keys
        )
        horizon = _scale_ceil(max_base + remaining_work + 1.0, scale)

        model = cp_model.CpModel()
        begin = {
            key: model.NewIntVar(current, horizon, f"begin_j{key[0]}_o{key[1]}")
            for key in operation_keys
        }
        completion = {
            key: model.NewIntVar(current, horizon, f"end_j{key[0]}_o{key[1]}")
            for key in operation_keys
        }
        assignment: dict[tuple[tuple[int, int], int], Any] = {}
        for key in operation_keys:
            operation = operation_by_key[key]
            vars_for_op = []
            for machine_id in operation.eligible_machine_ids:
                lit = model.NewBoolVar(f"x_j{key[0]}_o{key[1]}_m{machine_id}")
                assignment[(key, machine_id)] = lit
                vars_for_op.append(lit)
            model.AddExactlyOne(vars_for_op)

        for job in candidate_jobs:
            first = simulator.next_operation[job.job_id]
            first_key = (job.job_id, first)
            model.Add(
                begin[first_key]
                >= _scale_ceil(
                    max(simulator.current_time, simulator.operation_ready_at[job.job_id]),
                    scale,
                )
            )
            for operation_index in range(first + 1, len(job.operations)):
                previous = (job.job_id, operation_index - 1)
                current_key = (job.job_id, operation_index)
                model.Add(begin[current_key] >= completion[previous])

        setup_terms = []
        depot_arcs: dict[tuple[int, tuple[int, int]], Any] = {}
        for machine_id, machine in simulator.machines.items():
            machine_keys = [
                key for key in operation_keys if (key, machine_id) in assignment
            ]
            if not machine_keys:
                continue
            used = model.NewBoolVar(f"machine_{machine_id}_used")
            sum_x = sum(assignment[(key, machine_id)] for key in machine_keys)
            model.Add(sum_x >= used)
            model.Add(sum_x <= len(machine_keys) * used)

            arcs: list[tuple[int, int, Any]] = []
            depot_loop = model.NewBoolVar(f"m{machine_id}_depot_loop")
            model.Add(depot_loop + used == 1)
            arcs.append((0, 0, depot_loop))
            node_of = {key: index + 1 for index, key in enumerate(machine_keys)}

            for key in machine_keys:
                node = node_of[key]
                x = assignment[(key, machine_id)]
                self_loop = model.NewBoolVar(
                    f"m{machine_id}_self_j{key[0]}_o{key[1]}"
                )
                model.Add(self_loop + x == 1)
                arcs.append((node, node, self_loop))

                from_depot = model.NewBoolVar(
                    f"m{machine_id}_depot_to_j{key[0]}_o{key[1]}"
                )
                to_depot = model.NewBoolVar(
                    f"m{machine_id}_j{key[0]}_o{key[1]}_to_depot"
                )
                model.Add(from_depot <= x)
                model.Add(to_depot <= x)
                arcs.append((0, node, from_depot))
                arcs.append((node, 0, to_depot))
                depot_arcs[(machine_id, key)] = from_depot

                job = job_by_id[key[0]]
                processing = _scale_ceil(
                    operation_by_key[key].processing_time_on(machine_id), scale
                )
                setup = _scale_ceil(
                    simulator.setup_time(machine.last_family, job.family), scale
                )
                model.Add(begin[key] >= machine_available[machine_id]).OnlyEnforceIf(
                    from_depot
                )
                model.Add(completion[key] == begin[key] + processing + setup).OnlyEnforceIf(
                    from_depot
                )
                if setup:
                    setup_terms.append(setup * from_depot)

            for previous in machine_keys:
                previous_node = node_of[previous]
                previous_job = job_by_id[previous[0]]
                for next_key in machine_keys:
                    if previous == next_key:
                        continue
                    next_node = node_of[next_key]
                    next_job = job_by_id[next_key[0]]
                    arc = model.NewBoolVar(
                        f"m{machine_id}_j{previous[0]}o{previous[1]}"
                        f"_to_j{next_key[0]}o{next_key[1]}"
                    )
                    model.Add(arc <= assignment[(previous, machine_id)])
                    model.Add(arc <= assignment[(next_key, machine_id)])
                    arcs.append((previous_node, next_node, arc))
                    processing = _scale_ceil(
                        operation_by_key[next_key].processing_time_on(machine_id),
                        scale,
                    )
                    setup = _scale_ceil(
                        simulator.setup_time(previous_job.family, next_job.family),
                        scale,
                    )
                    model.Add(begin[next_key] >= completion[previous]).OnlyEnforceIf(arc)
                    model.Add(
                        completion[next_key] == begin[next_key] + processing + setup
                    ).OnlyEnforceIf(arc)
                    if setup:
                        setup_terms.append(setup * arc)

            model.AddCircuit(arcs)

        first_choice = []
        choice_by_action: dict[FJSPAction, Any] = {}
        for action in eligible_now:
            key = (action.job_id, action.operation_index)
            choice = model.NewBoolVar(
                f"first_j{action.job_id}_o{action.operation_index}_m{action.machine_id}"
            )
            choice_by_action[action] = choice
            first_choice.append(choice)
            model.Add(choice <= assignment[(key, action.machine_id)])
            model.Add(choice <= depot_arcs[(action.machine_id, key)])
            model.Add(begin[key] == current).OnlyEnforceIf(choice)
        model.AddExactlyOne(first_choice)

        tardiness_terms = []
        for job in candidate_jobs:
            final_key = (job.job_id, len(job.operations) - 1)
            tardiness = model.NewIntVar(0, horizon, f"tardiness_j{job.job_id}")
            due = int(round(job.due_date * scale))
            model.Add(tardiness >= completion[final_key] - due)
            tardiness_terms.append(job.priority * tardiness)

        makespan = model.NewIntVar(current, horizon, "makespan")
        model.AddMaxEquality(makespan, [completion[key] for key in operation_keys])
        total_setup = sum(setup_terms) if setup_terms else 0
        secondary_bound = horizon + _scale_ceil(max_setup, scale) * len(operation_keys)
        primary_weight = secondary_bound + 1
        model.Minimize(sum(tardiness_terms) * primary_weight + makespan + total_setup)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.solver_seconds
        solver.parameters.num_search_workers = self.config.num_search_workers
        solver.parameters.random_seed = self.config.random_seed
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        solve_time = float(solver.WallTime())
        candidate_ids = tuple(job.job_id for job in candidate_jobs)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = [
                action
                for action, choice in choice_by_action.items()
                if solver.BooleanValue(choice)
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f"CP-SAT returned {len(selected)} first actions; expected exactly one"
                )
            return FJSPCPSATDecision(
                action=selected[0],
                solver_status=status_name,
                objective_value=float(solver.ObjectiveValue()),
                solve_time_seconds=solve_time,
                fallback=False,
                candidate_job_ids=candidate_ids,
                candidate_operations=len(operation_keys),
            )
        if status == cp_model.UNKNOWN:
            fallback = earliest_due_date_action(simulator)
            return FJSPCPSATDecision(
                action=fallback,
                solver_status=status_name,
                objective_value=None,
                solve_time_seconds=solve_time,
                fallback=True,
                candidate_job_ids=candidate_ids,
                candidate_operations=len(operation_keys),
            )
        raise RuntimeError(f"FJSP CP-SAT failed with solver status {status_name}")


def _scale_ceil(value: float, scale: int) -> int:
    return int(ceil(max(value, 0.0) * scale - 1e-12))
