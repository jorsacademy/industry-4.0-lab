from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from time import perf_counter

from dmdtrl.env import DynamicManufacturingEnv, EnvConfig
from dmdtrl.or_policy import RollingHorizonCPSATPolicy


def run_cpsat_policy(
    policy: RollingHorizonCPSATPolicy,
    seed: int,
    config: EnvConfig | None = None,
) -> dict[str, float | int | str]:
    """Run one receding-horizon OR policy on one stochastic seed."""
    env = DynamicManufacturingEnv(config=config or EnvConfig())
    env.reset(seed=seed)
    terminated = False
    decision_times: list[float] = []
    fallback_decisions = 0
    total_decisions = 0

    while not terminated:
        started = perf_counter()
        decision = policy.choose(env)
        decision_times.append(perf_counter() - started)
        total_decisions += 1
        fallback_decisions += int(decision.used_fallback)

        _, _, terminated, truncated, _ = env.step_assignment(
            decision.job_id,
            decision.machine_id,
            decision_label=policy.name,
        )
        if truncated:
            break

    fallback_rate = fallback_decisions / total_decisions if total_decisions else 0.0
    return {
        "policy": policy.name,
        "seed": seed,
        **env.metrics(),
        "mean_decision_time_ms": 1_000.0 * mean(decision_times) if decision_times else 0.0,
        "solver_fallback_decisions": fallback_decisions,
        "solver_total_decisions": total_decisions,
        "solver_fallback_rate": fallback_rate,
        "solver_success_rate": 1.0 - fallback_rate,
    }


def evaluate_cpsat_policy(
    policy: RollingHorizonCPSATPolicy,
    seeds: Iterable[int],
    config: EnvConfig | None = None,
) -> list[dict[str, float | int | str]]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one evaluation seed is required")
    return [run_cpsat_policy(policy, seed, config) for seed in seed_list]
