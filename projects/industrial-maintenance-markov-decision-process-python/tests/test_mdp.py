import math
import unittest

import numpy as np

from industrial_maintenance_mdp import (
    CRITICAL,
    DEGRADED,
    FAILED,
    HEALTHY,
    MINOR,
    OPERATE,
    REPLACE,
    bellman_residual,
    default_maintenance_mdp,
    enumerate_valid_policies,
    evaluate_policy_exact,
    exhaustive_policy_oracle,
    monte_carlo_policy_evaluation,
    policy_iteration,
    policy_transition_matrix,
    q_values,
    sensitivity_analysis,
    stationary_distribution,
    value_iteration,
)


class IndustrialMaintenanceMDPTests(unittest.TestCase):
    def test_transition_rows_are_probabilities(self):
        model = default_maintenance_mdp()
        np.testing.assert_allclose(model.transition.sum(axis=2), 1.0, atol=1e-12)
        self.assertTrue(np.all(model.transition >= 0.0))

    def test_failed_state_only_allows_replacement(self):
        model = default_maintenance_mdp()
        self.assertFalse(model.allowed[FAILED, OPERATE])
        self.assertFalse(model.allowed[FAILED, MINOR])
        self.assertTrue(model.allowed[FAILED, REPLACE])

    def test_zero_value_bellman_backup_is_hand_checkable(self):
        model = default_maintenance_mdp()
        q = q_values(model, np.zeros(4))
        self.assertTrue(math.isclose(q[HEALTHY, OPERATE], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(q[DEGRADED, MINOR], 155.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(q[FAILED, REPLACE], 1270.0, abs_tol=1e-12))
        self.assertTrue(math.isinf(q[FAILED, OPERATE]))

    def test_three_exact_methods_agree(self):
        model = default_maintenance_mdp()
        vi = value_iteration(model)
        pi = policy_iteration(model)
        oracle = exhaustive_policy_oracle(model)
        expected = (OPERATE, MINOR, REPLACE, REPLACE)
        self.assertEqual(vi.policy, expected)
        self.assertEqual(pi.policy, expected)
        self.assertEqual(oracle.policy, expected)
        np.testing.assert_allclose(vi.value, oracle.value, atol=1e-7)
        np.testing.assert_allclose(pi.value, oracle.value, atol=1e-7)
        self.assertLess(bellman_residual(model, oracle.value), 1e-8)

    def test_exhaustive_oracle_checks_all_27_stationary_policies(self):
        model = default_maintenance_mdp()
        policies = list(enumerate_valid_policies(model))
        self.assertEqual(len(policies), 27)
        self.assertEqual(len(set(policies)), 27)
        self.assertEqual(exhaustive_policy_oracle(model).iterations, 27)

    def test_optimal_policy_dominates_two_baselines(self):
        model = default_maintenance_mdp()
        optimal = exhaustive_policy_oracle(model).value
        run_to_failure = evaluate_policy_exact(model, (OPERATE, OPERATE, OPERATE, REPLACE))
        critical_replace = evaluate_policy_exact(model, (OPERATE, OPERATE, REPLACE, REPLACE))
        self.assertTrue(np.all(optimal <= run_to_failure + 1e-9))
        self.assertTrue(np.all(optimal <= critical_replace + 1e-9))

    def test_stationary_distribution_is_invariant(self):
        model = default_maintenance_mdp()
        policy = exhaustive_policy_oracle(model).policy
        P = policy_transition_matrix(model, policy)
        pi = stationary_distribution(P)
        self.assertTrue(np.all(pi >= 0.0))
        self.assertTrue(math.isclose(float(pi.sum()), 1.0, abs_tol=1e-12))
        np.testing.assert_allclose(pi @ P, pi, atol=1e-10)

    def test_monte_carlo_is_reproducible(self):
        model = default_maintenance_mdp()
        policy = exhaustive_policy_oracle(model).policy
        a = monte_carlo_policy_evaluation(model, policy, trajectories_per_state=1000, horizon=100, seed=99)
        b = monte_carlo_policy_evaluation(model, policy, trajectories_per_state=1000, horizon=100, seed=99)
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.standard_error, b.standard_error)

    def test_failure_cost_sensitivity_changes_critical_action(self):
        rows = sensitivity_analysis(failure_state_costs=(400.0, 850.0))
        low_policy = rows[0][1]
        high_policy = rows[1][1]
        self.assertEqual(low_policy[CRITICAL], MINOR)
        self.assertEqual(high_policy[CRITICAL], REPLACE)
        self.assertEqual(low_policy[HEALTHY], OPERATE)
        self.assertEqual(high_policy[DEGRADED], MINOR)


if __name__ == "__main__":
    unittest.main()
