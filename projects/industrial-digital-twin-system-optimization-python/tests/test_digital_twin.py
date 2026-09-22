import math
import unittest

import numpy as np

from industrial_compressed_air_digital_twin import (
    brute_force_on_off_oracle,
    calibrate_digital_twin,
    default_compressors,
    default_demand_profile,
    default_electricity_price,
    generate_compressor_telemetry,
    max_dispatch_constraint_violation,
    optimize_system_dispatch,
    rule_based_baseline,
    simulate_dispatch_with_true_physics,
)


class IndustrialDigitalTwinTests(unittest.TestCase):
    def test_true_machine_power_relation(self):
        spec = default_compressors()[0]
        flow = 30.0
        expected = (
            spec.idle_power_kw
            + spec.marginal_power_kw_per_nm3_min
            * spec.degradation_factor
            * flow
        )
        self.assertTrue(math.isclose(
            spec.true_power_kw(flow, True),
            expected,
            abs_tol=1e-12,
        ))
        self.assertEqual(spec.true_power_kw(0.0, False), 0.0)

    def test_heldout_telemetry_recovers_degradation_slopes(self):
        specs = default_compressors()
        telemetry = generate_compressor_telemetry(
            specs,
            samples_per_compressor=300,
            seed=7,
            power_noise_std_kw=0.8,
        )
        calibrations = calibrate_digital_twin(specs, telemetry)

        for cal in calibrations:
            relative_error = abs(
                cal.estimated_slope_kw_per_flow
                - cal.true_slope_kw_per_flow
            ) / cal.true_slope_kw_per_flow
            self.assertLess(relative_error, 0.03)
            self.assertLess(cal.rmse_kw, 1.2)

    def test_tiny_milp_matches_bruteforce_binary_oracle(self):
        specs = default_compressors()[:2]
        telemetry = generate_compressor_telemetry(
            specs,
            samples_per_compressor=240,
            seed=13,
            power_noise_std_kw=0.7,
        )
        twins = tuple(
            c.twin
            for c in calibrate_digital_twin(specs, telemetry)
        )

        demand = np.array([48.0, 64.0])
        price = np.array([0.10, 0.20])

        result = optimize_system_dispatch(
            twins,
            demand,
            price,
            period_minutes=15.0,
            initial_storage_nm3=500.0,
            final_storage_min_nm3=450.0,
            storage_min_nm3=300.0,
            storage_max_nm3=700.0,
            initial_on=(0, 0),
            initial_flow_nm3_min=(0.0, 0.0),
        )
        oracle = brute_force_on_off_oracle(
            twins,
            demand,
            price,
            period_minutes=15.0,
            initial_storage_nm3=500.0,
            storage_min_nm3=300.0,
            storage_max_nm3=700.0,
            final_storage_min_nm3=450.0,
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(math.isclose(
            result.objective_cost,
            oracle,
            abs_tol=1e-7,
        ))

    def test_dispatch_postsolve_feasibility_audit(self):
        specs = default_compressors()
        telemetry = generate_compressor_telemetry(
            specs,
            samples_per_compressor=140,
            seed=21,
        )
        twins = tuple(
            c.twin
            for c in calibrate_digital_twin(specs, telemetry)
        )

        result = optimize_system_dispatch(
            twins,
            default_demand_profile(8),
            default_electricity_price(8),
            final_storage_min_nm3=700.0,
        )

        violation = max_dispatch_constraint_violation(
            twins,
            result,
            final_storage_min_nm3=700.0,
        )
        self.assertLess(violation, 1e-7)

    def test_true_physics_replay_closes_storage_balance(self):
        specs = default_compressors()
        telemetry = generate_compressor_telemetry(
            specs,
            samples_per_compressor=140,
            seed=33,
        )
        twins = tuple(
            c.twin
            for c in calibrate_digital_twin(specs, telemetry)
        )
        demand = default_demand_profile(8)
        price = default_electricity_price(8)

        dispatch = optimize_system_dispatch(
            twins,
            demand,
            price,
            final_storage_min_nm3=700.0,
        )
        validation = simulate_dispatch_with_true_physics(
            specs,
            dispatch,
            demand,
            price,
        )
        self.assertLess(validation.max_storage_balance_error, 1e-7)

    def test_rule_baseline_respects_machine_and_storage_constraints(self):
        specs = default_compressors()
        telemetry = generate_compressor_telemetry(
            specs,
            samples_per_compressor=140,
            seed=45,
        )
        twins = tuple(
            c.twin
            for c in calibrate_digital_twin(specs, telemetry)
        )

        baseline = rule_based_baseline(
            twins,
            default_demand_profile(),
            default_electricity_price(),
            storage_target_nm3=700.0,
        )
        self.assertLess(
            max_dispatch_constraint_violation(twins, baseline),
            1e-7,
        )

    def test_default_demand_and_price_profiles_have_expected_shape(self):
        demand = default_demand_profile()
        price = default_electricity_price()
        self.assertEqual(demand.shape, (24,))
        self.assertEqual(price.shape, (24,))
        self.assertTrue(np.all(demand > 0))
        self.assertTrue(np.all(price > 0))


if __name__ == "__main__":
    unittest.main()
