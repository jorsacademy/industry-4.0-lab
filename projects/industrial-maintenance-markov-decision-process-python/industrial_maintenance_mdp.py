from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

STATE_NAMES = ("Healthy", "Degraded", "Critical", "Failed")
ACTION_NAMES = ("Operate", "MinorMaintenance", "Replace")
HEALTHY, DEGRADED, CRITICAL, FAILED = range(4)
OPERATE, MINOR, REPLACE = range(3)


@dataclass(frozen=True)
class MDPModel:
    transition: np.ndarray  # [action, state, next_state]
    cost: np.ndarray        # [state, action]
    allowed: np.ndarray     # [state, action]
    discount: float = 0.97

    def __post_init__(self):
        if self.transition.shape != (3, 4, 4):
            raise ValueError("transition must be [3,4,4]")
        if self.cost.shape != (4, 3) or self.allowed.shape != (4, 3):
            raise ValueError("cost/allowed shape mismatch")
        if not 0 < self.discount < 1:
            raise ValueError("discount must be in (0,1)")
        if np.any(self.transition < 0) or not np.allclose(self.transition.sum(2), 1):
            raise ValueError("invalid transition probabilities")
        if np.any(self.cost < 0) or not np.all(self.allowed.any(1)):
            raise ValueError("invalid cost/action specification")


@dataclass(frozen=True)
class PolicySolution:
    policy: tuple[int, ...]
    value: np.ndarray
    iterations: int
    method: str


@dataclass(frozen=True)
class MonteCarloEstimate:
    mean: np.ndarray
    standard_error: np.ndarray
    trajectories_per_state: int
    horizon: int


def default_maintenance_mdp(
    *, discount: float = 0.97, failed_state_cost: float = 850.0,
    minor_maintenance_cost: float = 120.0, replacement_cost: float = 420.0,
) -> MDPModel:
    operate = np.array([
        [0.880, 0.105, 0.014, 0.001],
        [0.000, 0.730, 0.220, 0.050],
        [0.000, 0.000, 0.520, 0.480],
        [0.000, 0.000, 0.000, 1.000],
    ])
    minor = np.array([
        [0.970, 0.028, 0.002, 0.000],
        [0.680, 0.290, 0.028, 0.002],
        [0.080, 0.600, 0.270, 0.050],
        [0.000, 0.000, 0.000, 1.000],
    ])
    replace = np.tile(np.array([0.995, 0.005, 0.0, 0.0]), (4, 1))
    transition = np.stack([operate, minor, replace])

    state_cost = np.array([0.0, 35.0, 160.0, failed_state_cost])
    action_cost = np.array([0.0, minor_maintenance_cost, replacement_cost])
    cost = state_cost[:, None] + action_cost[None, :]

    allowed = np.ones((4, 3), dtype=bool)
    allowed[FAILED] = False
    allowed[FAILED, REPLACE] = True
    return MDPModel(transition, cost, allowed, discount)


def q_values(model: MDPModel, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (4,):
        raise ValueError("value must have length 4")
    q = np.full((4, 3), np.inf)
    for s in range(4):
        for a in range(3):
            if model.allowed[s, a]:
                q[s, a] = model.cost[s, a] + model.discount * (model.transition[a, s] @ value)
    return q


def bellman_residual(model: MDPModel, value: np.ndarray) -> float:
    return float(np.max(np.abs(value - np.min(q_values(model, value), axis=1))))


def value_iteration(model: MDPModel, *, tolerance: float = 1e-11, max_iterations: int = 100_000) -> PolicySolution:
    value = np.zeros(4)
    for iteration in range(1, max_iterations + 1):
        updated = np.min(q_values(model, value), axis=1)
        if np.max(np.abs(updated - value)) <= tolerance:
            policy = tuple(map(int, np.argmin(q_values(model, updated), axis=1)))
            return PolicySolution(policy, updated, iteration, "value_iteration")
        value = updated
    raise RuntimeError("value iteration did not converge")


def evaluate_policy_exact(model: MDPModel, policy: Sequence[int]) -> np.ndarray:
    policy = tuple(map(int, policy))
    if len(policy) != 4:
        raise ValueError("policy must have four actions")
    for s, a in enumerate(policy):
        if a not in range(3) or not model.allowed[s, a]:
            raise ValueError(f"invalid action {a} in state {STATE_NAMES[s]}")
    P = np.vstack([model.transition[policy[s], s] for s in range(4)])
    c = np.array([model.cost[s, policy[s]] for s in range(4)])
    return np.linalg.solve(np.eye(4) - model.discount * P, c)


def policy_iteration(model: MDPModel, *, max_iterations: int = 1000) -> PolicySolution:
    policy = tuple(int(np.flatnonzero(model.allowed[s])[0]) for s in range(4))
    for iteration in range(1, max_iterations + 1):
        value = evaluate_policy_exact(model, policy)
        improved = tuple(map(int, np.argmin(q_values(model, value), axis=1)))
        if improved == policy:
            return PolicySolution(policy, value, iteration, "policy_iteration")
        policy = improved
    raise RuntimeError("policy iteration did not converge")


def enumerate_valid_policies(model: MDPModel) -> Iterable[tuple[int, ...]]:
    return itertools.product(*(tuple(np.flatnonzero(model.allowed[s])) for s in range(4)))


def exhaustive_policy_oracle(model: MDPModel, *, tolerance: float = 1e-9) -> PolicySolution:
    records = [(tuple(map(int, p)), evaluate_policy_exact(model, p)) for p in enumerate_valid_policies(model)]
    componentwise_best = np.min(np.vstack([v for _, v in records]), axis=0)
    candidates = [(p, v) for p, v in records if np.all(v <= componentwise_best + tolerance)]
    if not candidates:
        raise RuntimeError("no componentwise-optimal stationary policy")
    policy, value = min(candidates, key=lambda item: item[0])
    return PolicySolution(policy, value, len(records), "exhaustive_policy_oracle")


def policy_transition_matrix(model: MDPModel, policy: Sequence[int]) -> np.ndarray:
    policy = tuple(map(int, policy))
    evaluate_policy_exact(model, policy)
    return np.vstack([model.transition[policy[s], s] for s in range(4)])


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.shape != (4, 4) or not np.allclose(P.sum(1), 1):
        raise ValueError("P must be a 4x4 stochastic matrix")
    A = P.T - np.eye(4)
    A[-1] = 1.0
    b = np.zeros(4); b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None); pi /= pi.sum()
    return pi


def stationary_policy_diagnostics(model: MDPModel, policy: Sequence[int]) -> tuple[float, float]:
    policy = tuple(map(int, policy))
    pi = stationary_distribution(policy_transition_matrix(model, policy))
    costs = np.array([model.cost[s, policy[s]] for s in range(4)])
    return float(pi @ costs), float(pi[FAILED])


def monte_carlo_policy_evaluation(
    model: MDPModel, policy: Sequence[int], *, trajectories_per_state: int = 20_000,
    horizon: int = 350, seed: int = 42,
) -> MonteCarloEstimate:
    policy = tuple(map(int, policy)); evaluate_policy_exact(model, policy)
    if trajectories_per_state < 100 or horizon < 10:
        raise ValueError("insufficient Monte Carlo effort")
    rng = np.random.default_rng(seed)
    means = np.zeros(4); ses = np.zeros(4)
    for start in range(4):
        states = np.full(trajectories_per_state, start, dtype=np.int8)
        returns = np.zeros(trajectories_per_state)
        discount_power = 1.0
        for _ in range(horizon):
            actions = np.take(np.array(policy, dtype=np.int8), states)
            returns += discount_power * model.cost[states, actions]
            u = rng.random(trajectories_per_state)
            next_states = np.empty_like(states)
            for s in range(4):
                mask = states == s
                if np.any(mask):
                    next_states[mask] = np.searchsorted(np.cumsum(model.transition[policy[s], s]), u[mask], side="right")
            states = next_states
            discount_power *= model.discount
        means[start] = returns.mean()
        ses[start] = returns.std(ddof=1) / math.sqrt(trajectories_per_state)
    return MonteCarloEstimate(means, ses, trajectories_per_state, horizon)


def sensitivity_analysis(failure_state_costs=(400.0, 600.0, 850.0, 1100.0, 1400.0), *, discount=0.97):
    rows = []
    for c in failure_state_costs:
        model = default_maintenance_mdp(discount=discount, failed_state_cost=float(c))
        sol = exhaustive_policy_oracle(model)
        rows.append((float(c), sol.policy, float(sol.value[HEALTHY])))
    return rows


def run_end_to_end(*, discount=0.97, monte_carlo_trajectories=20_000, monte_carlo_horizon=350, seed=42):
    model = default_maintenance_mdp(discount=discount)
    vi = value_iteration(model); pi = policy_iteration(model); ex = exhaustive_policy_oracle(model)
    if not (vi.policy == pi.policy == ex.policy):
        raise RuntimeError("exact methods disagree on policy")
    if not np.allclose(vi.value, ex.value, atol=1e-7) or not np.allclose(pi.value, ex.value, atol=1e-7):
        raise RuntimeError("exact methods disagree on value")
    residual = bellman_residual(model, ex.value)
    if residual > 1e-8:
        raise RuntimeError("Bellman residual too large")

    mc = monte_carlo_policy_evaluation(model, ex.policy, trajectories_per_state=monte_carlo_trajectories,
                                       horizon=monte_carlo_horizon, seed=seed)
    max_allowed_cost = float(np.max(model.cost[model.allowed]))
    tail_bound = model.discount ** monte_carlo_horizon * max_allowed_cost / (1 - model.discount)
    permitted = 5 * mc.standard_error + tail_bound + 1.0
    if np.any(np.abs(mc.mean - ex.value) > permitted):
        raise RuntimeError("Monte Carlo check failed")
    return model, vi, pi, ex, mc, residual


def self_test():
    model = default_maintenance_mdp()
    assert np.allclose(model.transition.sum(2), 1)
    q0 = q_values(model, np.zeros(4))
    assert math.isclose(q0[HEALTHY, OPERATE], 0)
    assert math.isclose(q0[DEGRADED, MINOR], 155)
    assert math.isclose(q0[FAILED, REPLACE], 1270)
    assert math.isinf(q0[FAILED, OPERATE])

    expected = (OPERATE, MINOR, REPLACE, REPLACE)
    vi = value_iteration(model); pi = policy_iteration(model); ex = exhaustive_policy_oracle(model)
    assert vi.policy == pi.policy == ex.policy == expected
    assert np.allclose(vi.value, ex.value, atol=1e-7)
    assert np.allclose(pi.value, ex.value, atol=1e-7)
    assert bellman_residual(model, ex.value) < 1e-8
    assert len(list(enumerate_valid_policies(model))) == 27

    P = policy_transition_matrix(model, expected)
    stationary = stationary_distribution(P)
    assert np.all(stationary >= 0) and math.isclose(stationary.sum(), 1.0)
    assert np.allclose(stationary @ P, stationary, atol=1e-10)

    run_to_failure = evaluate_policy_exact(model, (OPERATE, OPERATE, OPERATE, REPLACE))
    critical_replace = evaluate_policy_exact(model, (OPERATE, OPERATE, REPLACE, REPLACE))
    assert np.all(ex.value <= run_to_failure + 1e-9)
    assert np.all(ex.value <= critical_replace + 1e-9)

    a = monte_carlo_policy_evaluation(model, expected, trajectories_per_state=1000, horizon=100, seed=7)
    b = monte_carlo_policy_evaluation(model, expected, trajectories_per_state=1000, horizon=100, seed=7)
    assert np.array_equal(a.mean, b.mean)
    print("Industrial maintenance MDP self-test: OK")


def print_result(result):
    model, vi, pi, ex, mc, residual = result
    print("=" * 78)
    print("CONDITION-BASED INDUSTRIAL MAINTENANCE — MARKOV DECISION PROCESS")
    print("=" * 78)
    print("Optimal stationary policy")
    for s, a in zip(STATE_NAMES, ex.policy):
        print(f"  {s:<10} -> {ACTION_NAMES[a]}")
    print("\nDiscounted value by initial state [$1,000]")
    for s, v in zip(STATE_NAMES, ex.value):
        print(f"  {s:<10}: {v:10.3f}")
    print(f"\nValue-iteration iterations   : {vi.iterations}")
    print(f"Policy-iteration iterations  : {pi.iterations}")
    print(f"Policies exhaustively checked: {ex.iterations}")
    print(f"Bellman residual             : {residual:.3e}")
    print("\nMonte Carlo cross-check")
    for s, mean, se, exact in zip(STATE_NAMES, mc.mean, mc.standard_error, ex.value):
        print(f"  {s:<10}: MC={mean:10.3f} ± {1.96*se:7.3f} (exact={exact:10.3f})")

    baselines = {
        "Optimal condition-based": ex.policy,
        "Run-to-failure": (OPERATE, OPERATE, OPERATE, REPLACE),
        "Critical-state replacement": (OPERATE, OPERATE, REPLACE, REPLACE),
    }
    print("\nPolicy comparison")
    for name, policy in baselines.items():
        value = evaluate_policy_exact(model, policy)
        avg, p_fail = stationary_policy_diagnostics(model, policy)
        print(f"  {name:<27} V(Healthy)={value[0]:9.3f} avg={avg:7.3f} P(Failed)={100*p_fail:6.3f}%")
    print("\nExactness applies to the declared finite discounted MDP; model parameters are stylized.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--discount", type=float, default=0.97)
    p.add_argument("--monte-carlo-trajectories", type=int, default=20_000)
    p.add_argument("--monte-carlo-horizon", type=int, default=350)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        print_result(run_end_to_end(discount=args.discount,
                                    monte_carlo_trajectories=args.monte_carlo_trajectories,
                                    monte_carlo_horizon=args.monte_carlo_horizon,
                                    seed=args.seed))
