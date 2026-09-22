from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dmdtrl.fjsp_generator import FJSPGeneratorConfig, generate_fjsp_instance
from dmdtrl.fjsp_models import FJSPAction
from dmdtrl.fjsp_simulator import FlexibleJobShopSimulator


@dataclass(slots=True, frozen=True)
class FJSPActionCodec:
    job_ids: tuple[int, ...]
    max_operations: int
    n_machines: int

    def __post_init__(self) -> None:
        if not self.job_ids:
            raise ValueError("job_ids must not be empty")
        if len(set(self.job_ids)) != len(self.job_ids):
            raise ValueError("job_ids must be unique")
        if self.max_operations <= 0 or self.n_machines <= 0:
            raise ValueError("max_operations and n_machines must be positive")

    @property
    def size(self) -> int:
        return len(self.job_ids) * self.max_operations * self.n_machines

    def encode(self, action: FJSPAction) -> int:
        try:
            job_slot = self.job_ids.index(action.job_id)
        except ValueError as exc:
            raise ValueError(f"unknown job_id {action.job_id}") from exc
        if not 0 <= action.operation_index < self.max_operations:
            raise ValueError("operation_index is outside codec capacity")
        if not 0 <= action.machine_id < self.n_machines:
            raise ValueError("machine_id is outside codec capacity")
        return (
            (job_slot * self.max_operations + action.operation_index) * self.n_machines
            + action.machine_id
        )

    def decode(self, action_id: int) -> FJSPAction:
        action_id = int(action_id)
        if not 0 <= action_id < self.size:
            raise ValueError(f"action_id {action_id} is outside [0, {self.size})")
        job_operation_slot, machine_id = divmod(action_id, self.n_machines)
        job_slot, operation_index = divmod(job_operation_slot, self.max_operations)
        return FJSPAction(
            job_id=self.job_ids[job_slot],
            operation_index=operation_index,
            machine_id=machine_id,
        )


@dataclass(slots=True)
class FJSPEnvConfig:
    generator: FJSPGeneratorConfig = field(default_factory=FJSPGeneratorConfig)
    default_setup_time: float = 1.0
    operation_bonus: float = 0.05
    job_completion_bonus: float = 1.0
    waiting_weight: float = 0.02
    setup_weight: float = 0.05
    tardiness_weight: float = 0.20

    def validate(self) -> None:
        self.generator.validate()
        if self.default_setup_time < 0.0:
            raise ValueError("default_setup_time must be non-negative")
        for name in (
            "operation_bonus",
            "job_completion_bonus",
            "waiting_weight",
            "setup_weight",
            "tardiness_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


class FlexibleJobShopEnv(gym.Env[np.ndarray, int]):
    """Gymnasium FJSP environment with a fixed action index and dynamic mask."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: FJSPEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FJSPEnvConfig()
        self.config.validate()
        generator = self.config.generator
        self.codec = FJSPActionCodec(
            job_ids=tuple(range(generator.n_jobs)),
            max_operations=generator.operations_max,
            n_machines=generator.n_machines,
        )
        self.action_space = spaces.Discrete(self.codec.size)
        self._job_feature_count = 9 + 2 * generator.n_machines
        self._machine_feature_count = 5
        observation_size = (
            7
            + generator.n_jobs * self._job_feature_count
            + generator.n_machines * self._machine_feature_count
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.render_mode = render_mode
        self.simulator: FlexibleJobShopSimulator | None = None
        self.decision_trace: list[dict[str, float | int | bool]] = []
        self._time_scale = 1.0
        self._max_processing = 1.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        instance = generate_fjsp_instance(self.np_random, self.config.generator)
        self.simulator = FlexibleJobShopSimulator(
            instance,
            default_setup_time=self.config.default_setup_time,
        )
        self.decision_trace = []
        self._max_processing = max(
            option.processing_time
            for job in instance.jobs
            for operation in job.operations
            for option in operation.machine_options
        )
        self._time_scale = max(
            max(job.due_date for job in instance.jobs),
            sum(
                min(option.processing_time for option in operation.machine_options)
                for job in instance.jobs
                for operation in job.operations
            )
            / instance.n_machines,
            1.0,
        )
        return self._observation(), self._info()

    def action_masks(self) -> np.ndarray:
        simulator = self._simulator()
        mask = np.zeros(self.codec.size, dtype=bool)
        for action in simulator.eligible_actions():
            mask[self.codec.encode(action)] = True
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        simulator = self._simulator()
        action_id = int(action)
        decoded = self.codec.decode(action_id)
        mask = self.action_masks()
        if not bool(mask[action_id]):
            raise ValueError(f"action_id {action_id} decodes to infeasible action {decoded}")

        decision_time = simulator.current_time
        feasible_count = int(mask.sum())
        job = simulator.job(decoded.job_id)
        final_operation = decoded.operation_index == len(job.operations) - 1
        simulator.step(decoded)
        scheduled = simulator.schedule[-1]
        weighted_tardiness = 0.0
        if final_operation:
            weighted_tardiness = job.priority * max(0.0, scheduled.completion_time - job.due_date)

        reward = (
            self.config.operation_bonus
            + (self.config.job_completion_bonus if final_operation else 0.0)
            - self.config.waiting_weight * scheduled.waiting_time
            - self.config.setup_weight * scheduled.setup_time
            - self.config.tardiness_weight * weighted_tardiness
        )
        self.decision_trace.append(
            {
                "decision_index": len(self.decision_trace),
                "decision_time": float(decision_time),
                "action_id": action_id,
                "job_id": decoded.job_id,
                "operation_index": decoded.operation_index,
                "machine_id": decoded.machine_id,
                "feasible_action_count": feasible_count,
                "final_operation": final_operation,
                "completion_time": float(scheduled.completion_time),
                "reward": float(reward),
            }
        )
        terminated = simulator.terminated
        return self._observation(), float(reward), terminated, False, self._info()

    def _simulator(self) -> FlexibleJobShopSimulator:
        if self.simulator is None:
            raise RuntimeError("environment is not reset")
        return self.simulator

    def _observation(self) -> np.ndarray:
        simulator = self._simulator()
        cfg = self.config.generator
        current_time = simulator.current_time
        actions = simulator.eligible_actions()
        metrics = simulator.metrics()
        unfinished = [
            job
            for job in simulator.instance.jobs
            if simulator.next_operation[job.job_id] < len(job.operations)
        ]
        slacks = [self._job_slack(job.job_id) for job in unfinished]
        overdue_fraction = (
            sum(current_time > job.due_date for job in unfinished) / max(len(unfinished), 1)
        )
        mean_slack = float(np.mean(slacks)) if slacks else 0.0
        slack_scale = max(self._max_processing * cfg.operations_max, 1.0)
        utilization = metrics["utilization"]

        values: list[float] = [
            current_time / max(self._time_scale, current_time, 1.0),
            len(simulator.schedule) / simulator.instance.total_operations,
            len(simulator.job_completion_times) / len(simulator.instance.jobs),
            len(actions) / max(self.codec.size, 1),
            min(1.0, utilization),
            0.5 + 0.5 * np.tanh(mean_slack / slack_scale),
            overdue_fraction,
        ]

        for job_id in self.codec.job_ids:
            job = simulator.job(job_id)
            next_index = simulator.next_operation[job_id]
            completed = next_index >= len(job.operations)
            released = current_time + 1e-12 >= job.release_time
            ready = (not completed) and simulator.operation_ready_at[job_id] <= current_time + 1e-12
            progress = next_index / len(job.operations)
            family_norm = (job.family + 1) / (cfg.n_families + 1)
            slack_scaled = 0.5 + 0.5 * np.tanh(self._job_slack(job_id) / slack_scale)
            next_index_norm = min(next_index, cfg.operations_max) / cfg.operations_max
            values.extend(
                [
                    1.0,
                    float(released),
                    float(completed),
                    progress,
                    min(job.priority / 3.0, 1.0),
                    family_norm,
                    slack_scaled,
                    float(ready),
                    next_index_norm,
                ]
            )

            eligible = np.zeros(cfg.n_machines, dtype=float)
            processing = np.zeros(cfg.n_machines, dtype=float)
            if not completed:
                operation = job.operations[next_index]
                for option in operation.machine_options:
                    eligible[option.machine_id] = 1.0
                    processing[option.machine_id] = option.processing_time / self._max_processing
            values.extend(eligible.tolist())
            values.extend(processing.tolist())

        for machine_id in range(cfg.n_machines):
            machine = simulator.machines[machine_id]
            available = machine.available_at <= current_time + 1e-12
            time_until_available = max(0.0, machine.available_at - current_time)
            family_norm = (
                0.0
                if machine.last_family is None
                else (machine.last_family + 1) / (cfg.n_families + 1)
            )
            values.extend(
                [
                    float(available),
                    time_until_available / max(self._time_scale, time_until_available, 1.0),
                    family_norm,
                    machine.busy_time / max(self._time_scale, machine.busy_time, 1.0),
                    machine.setup_time / max(self._time_scale, machine.setup_time, 1.0),
                ]
            )

        observation = np.asarray(values, dtype=np.float32)
        return np.clip(observation, 0.0, 1.0)

    def _job_slack(self, job_id: int) -> float:
        simulator = self._simulator()
        job = simulator.job(job_id)
        next_index = simulator.next_operation[job_id]
        if next_index >= len(job.operations):
            completion = simulator.job_completion_times[job_id]
            return job.due_date - completion
        remaining = sum(
            min(option.processing_time for option in operation.machine_options)
            for operation in job.operations[next_index:]
        )
        ready_time = max(simulator.operation_ready_at[job_id], simulator.current_time)
        return job.due_date - ready_time - remaining

    def _info(self) -> dict[str, Any]:
        simulator = self._simulator()
        return {
            "time": float(simulator.current_time),
            "feasible_action_count": int(self.action_masks().sum()),
            "metrics": simulator.metrics(),
            "decision_count": len(self.decision_trace),
        }

    def decision_trace_records(self) -> list[dict[str, float | int | bool]]:
        return [dict(record) for record in self.decision_trace]

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self.simulator is None:
            return None
        metrics = self.simulator.metrics()
        return (
            f"t={self.simulator.current_time:.2f} "
            f"ops={len(self.simulator.schedule)}/{self.simulator.instance.total_operations} "
            f"feasible={int(self.action_masks().sum())} "
            f"WTT={metrics['weighted_tardiness']:.2f}"
        )
