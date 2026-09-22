from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dmdtrl.dispatch import DispatchRule, rule_from_action, select_job
from dmdtrl.generator import generate_jobs, generate_machines
from dmdtrl.models import Job, Machine, ScheduledOperation


@dataclass(slots=True)
class EnvConfig:
    n_jobs: int = 60
    n_machines: int = 4
    n_families: int = 5
    mean_interarrival: float = 1.5
    processing_min: float = 2.0
    processing_max: float = 8.0
    machine_speed_min: float = 0.85
    machine_speed_max: float = 1.15
    due_date_factor_min: float = 1.5
    due_date_factor_max: float = 3.0
    sequence_setup_time: float = 1.5
    breakdown_probability: float = 0.04
    repair_time_min: float = 2.0
    repair_time_max: float = 7.0
    waiting_weight: float = 0.05
    tardiness_weight: float = 0.30
    setup_weight: float = 0.10
    repair_weight: float = 0.10
    quality_weight: float = 0.05
    completion_bonus: float = 1.0
    on_time_bonus: float = 0.25


class DynamicManufacturingEnv(gym.Env[np.ndarray, int]):
    """Event-driven digital twin for dynamic parallel-machine scheduling.

    The RL action selects one of eight dispatching rules. Research baselines
    that make explicit job-machine decisions can use ``step_assignment``;
    both paths execute through the same stochastic transition logic.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        if self.config.n_jobs <= 0 or self.config.n_machines <= 0:
            raise ValueError("n_jobs and n_machines must be positive")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(DispatchRule))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)

        self.jobs: list[Job] = []
        self.machines: list[Machine] = []
        self.queue: list[Job] = []
        self.schedule: list[ScheduledOperation] = []
        self.current_time = 0.0
        self._next_job_idx = 0
        self._completed_job_ids: set[int] = set()
        self._total_waiting = 0.0
        self._total_tardiness = 0.0
        self._weighted_tardiness = 0.0
        self._total_setup = 0.0
        self._total_repair = 0.0
        self._quality_risk_processed = 0.0
        self._max_time_scale = 1.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        self.jobs = generate_jobs(
            self.np_random,
            n_jobs=cfg.n_jobs,
            n_families=cfg.n_families,
            mean_interarrival=cfg.mean_interarrival,
            processing_range=(cfg.processing_min, cfg.processing_max),
            due_date_factor_range=(cfg.due_date_factor_min, cfg.due_date_factor_max),
        )
        self.machines = generate_machines(
            self.np_random,
            n_machines=cfg.n_machines,
            speed_range=(cfg.machine_speed_min, cfg.machine_speed_max),
        )
        self.queue = []
        self.schedule = []
        self.current_time = 0.0
        self._next_job_idx = 0
        self._completed_job_ids = set()
        self._total_waiting = 0.0
        self._total_tardiness = 0.0
        self._weighted_tardiness = 0.0
        self._total_setup = 0.0
        self._total_repair = 0.0
        self._quality_risk_processed = 0.0
        latest_due = max(job.due_date for job in self.jobs)
        self._max_time_scale = max(latest_due, cfg.n_jobs * cfg.mean_interarrival, 1.0)
        self._advance_to_decision()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._ensure_schedulable()
        rule = rule_from_action(action)
        machine = min(self._available_machines(), key=lambda m: (m.available_at, m.machine_id))
        job = select_job(
            self.queue,
            machine=machine,
            current_time=self.current_time,
            rule=rule,
            setup_time=self.config.sequence_setup_time,
        )
        return self._execute_assignment(job, machine, rule.name)

    def step_assignment(
        self,
        job_id: int,
        machine_id: int,
        *,
        decision_label: str = "EXTERNAL",
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute an explicit released-job / currently-available-machine decision."""
        self._ensure_schedulable()
        try:
            job = next(job for job in self.queue if job.job_id == int(job_id))
        except StopIteration as exc:
            raise ValueError(f"job_id {job_id} is not in the released queue") from exc

        available = {machine.machine_id: machine for machine in self._available_machines()}
        try:
            machine = available[int(machine_id)]
        except KeyError as exc:
            raise ValueError(f"machine_id {machine_id} is not currently available") from exc

        return self._execute_assignment(job, machine, decision_label)

    def queued_jobs(self) -> tuple[Job, ...]:
        """Return an immutable snapshot of jobs released at the current decision epoch."""
        return tuple(self.queue)

    def available_machines(self) -> tuple[Machine, ...]:
        """Return an immutable snapshot of machines available at the current decision epoch."""
        return tuple(self._available_machines())

    def _ensure_schedulable(self) -> None:
        if len(self._completed_job_ids) >= self.config.n_jobs:
            raise RuntimeError("episode is already terminated; call reset()")
        if not self.queue:
            self._advance_to_decision()
        if not self.queue:
            raise RuntimeError("no schedulable job available")

    def _execute_assignment(
        self,
        job: Job,
        machine: Machine,
        decision_label: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.queue.remove(job)

        start = max(self.current_time, machine.available_at, job.arrival_time)
        setup = 0.0 if machine.last_family in (None, job.family) else self.config.sequence_setup_time
        repair = 0.0
        if self.np_random.random() < self.config.breakdown_probability:
            repair = float(
                self.np_random.uniform(self.config.repair_time_min, self.config.repair_time_max)
            )

        processing = job.processing_time / machine.speed
        completion = start + repair + setup + processing
        waiting = max(0.0, start - job.arrival_time)
        tardiness = max(0.0, completion - job.due_date)
        weighted_tardiness = job.priority * tardiness
        on_time = tardiness <= 1e-9

        machine.available_at = completion
        machine.last_family = job.family
        machine.busy_time += processing
        machine.setup_time += setup
        machine.repair_time += repair

        op = ScheduledOperation(
            job_id=job.job_id,
            machine_id=machine.machine_id,
            family=job.family,
            start_time=start,
            completion_time=completion,
            processing_time=processing,
            setup_time=setup,
            repair_time=repair,
            waiting_time=waiting,
            tardiness=tardiness,
            weighted_tardiness=weighted_tardiness,
            on_time=on_time,
            rule=decision_label,
        )
        self.schedule.append(op)
        self._completed_job_ids.add(job.job_id)
        self._total_waiting += waiting
        self._total_tardiness += tardiness
        self._weighted_tardiness += weighted_tardiness
        self._total_setup += setup
        self._total_repair += repair
        self._quality_risk_processed += job.quality_risk

        reward = (
            self.config.completion_bonus
            + (self.config.on_time_bonus if on_time else 0.0)
            - self.config.waiting_weight * waiting
            - self.config.tardiness_weight * weighted_tardiness
            - self.config.setup_weight * setup
            - self.config.repair_weight * repair
            - self.config.quality_weight * job.quality_risk
        )

        terminated = len(self._completed_job_ids) == self.config.n_jobs
        if not terminated:
            self._advance_to_decision()
        else:
            self.current_time = max(m.available_at for m in self.machines)

        return self._observation(), float(reward), terminated, False, self._info()

    def _release_jobs(self) -> None:
        while self._next_job_idx < len(self.jobs):
            job = self.jobs[self._next_job_idx]
            if job.arrival_time > self.current_time + 1e-12:
                break
            self.queue.append(job)
            self._next_job_idx += 1

    def _available_machines(self) -> list[Machine]:
        return [m for m in self.machines if m.available_at <= self.current_time + 1e-12]

    def _advance_to_decision(self) -> None:
        """Advance simulated time until both a job and a machine are available."""
        while len(self._completed_job_ids) < self.config.n_jobs:
            self._release_jobs()
            if self.queue and self._available_machines():
                return

            candidates: list[float] = []
            if self._next_job_idx < len(self.jobs):
                next_arrival = self.jobs[self._next_job_idx].arrival_time
                if next_arrival > self.current_time + 1e-12:
                    candidates.append(next_arrival)
            future_machine_times = [
                m.available_at for m in self.machines if m.available_at > self.current_time + 1e-12
            ]
            candidates.extend(future_machine_times)
            if not candidates:
                return
            self.current_time = min(candidates)

    def _observation(self) -> np.ndarray:
        cfg = self.config
        queue_len = len(self.queue)
        workloads = np.array([j.processing_time for j in self.queue], dtype=float)
        waits = np.array(
            [max(0.0, self.current_time - j.arrival_time) for j in self.queue], dtype=float
        )
        slacks = np.array(
            [j.due_date - self.current_time - j.processing_time for j in self.queue], dtype=float
        )
        overdue = np.array([self.current_time > j.due_date for j in self.queue], dtype=float)
        urgent = np.array([j.priority == 3 for j in self.queue], dtype=float)
        risks = np.array([j.quality_risk for j in self.queue], dtype=float)

        available = self._available_machines()
        same_family = 0.0
        if self.queue and available:
            opportunities = [
                any(m.last_family is not None and m.last_family == j.family for m in available)
                for j in self.queue
            ]
            same_family = float(np.mean(opportunities))

        horizon = max(self.current_time, 1.0)
        utilization = sum(m.busy_time for m in self.machines) / (cfg.n_machines * horizon)
        disruption = sum(m.repair_time for m in self.machines) / (cfg.n_machines * horizon)
        setup_load = sum(m.setup_time for m in self.machines) / (cfg.n_machines * horizon)

        max_proc = max(cfg.processing_max, 1.0)
        max_wait = max(self._max_time_scale, 1.0)
        mean_slack = float(np.mean(slacks)) if slacks.size else 0.0
        slack_scaled = 0.5 + 0.5 * np.tanh(mean_slack / max_proc)

        obs = np.array(
            [
                queue_len / cfg.n_jobs,
                (float(np.sum(workloads)) if workloads.size else 0.0) / (cfg.n_jobs * max_proc),
                (float(np.mean(waits)) if waits.size else 0.0) / max_wait,
                float(np.mean(urgent)) if urgent.size else 0.0,
                same_family,
                slack_scaled,
                float(np.mean(overdue)) if overdue.size else 0.0,
                (float(np.mean(workloads)) if workloads.size else 0.0) / max_proc,
                (float(np.std(workloads)) if workloads.size else 0.0) / max_proc,
                float(np.mean(risks)) if risks.size else 0.0,
                utilization,
                self.current_time / max(self._max_time_scale, self.current_time, 1.0),
                len(self._completed_job_ids) / cfg.n_jobs,
                min(1.0, disruption + setup_load),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def metrics(self) -> dict[str, float]:
        completed = len(self.schedule)
        on_time = sum(op.on_time for op in self.schedule)
        makespan = max((op.completion_time for op in self.schedule), default=0.0)
        total_busy = sum(m.busy_time for m in self.machines)
        utilization = total_busy / max(self.config.n_machines * makespan, 1e-9)
        return {
            "completed_jobs": float(completed),
            "makespan": float(makespan),
            "mean_waiting_time": self._total_waiting / max(completed, 1),
            "total_tardiness": float(self._total_tardiness),
            "weighted_tardiness": float(self._weighted_tardiness),
            "total_setup_time": float(self._total_setup),
            "total_repair_time": float(self._total_repair),
            "on_time_rate": on_time / max(completed, 1),
            "utilization": float(utilization),
            "mean_quality_risk": self._quality_risk_processed / max(completed, 1),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "time": float(self.current_time),
            "queue_length": len(self.queue),
            "completed_jobs": len(self._completed_job_ids),
            "metrics": self.metrics(),
        }

    def schedule_records(self) -> list[dict[str, Any]]:
        return [asdict(op) for op in self.schedule]

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        metrics = self.metrics()
        return (
            f"t={self.current_time:.2f} queue={len(self.queue)} "
            f"completed={len(self.schedule)}/{self.config.n_jobs} "
            f"WTT={metrics['weighted_tardiness']:.2f}"
        )
