import math
import unittest

import numpy as np

from industrial_compressed_air_digital_twin import (
    TelemetryBatch,
    default_compressors,
    generate_compressor_telemetry,
    optimize_system_dispatch,
)
from physics_informed_surrogate import (
    benchmark_sparse_calibration,
    fit_data_only_power_surrogate,
    fit_physics_informed_power_surrogate,
)


class PhysicsInformedSurrogateTests(unittest.TestCase):
    def test_physics_informed_surrogate_enforces_off_state_zero_and_positive_parameters(self):
        spec = default_compressors()[0]
        telemetry = generate_compressor_telemetry(
            [spec],
            samples_per_compressor=80,
            seed=12,
            power_noise_std_kw=2.0,
        )[0]
        model = fit_physics_informed_power_surrogate(spec, telemetry)
        self.assertGreaterEqual(model.idle_power_kw, 0.0)
        self.assertGreaterEqual(model.degradation_factor, 0.0)
        prediction = model.predict(
            np.array([0.0, spec.min_flow_nm3_min]),
            np.array([False, True]),
        )
        self.assertEqual(float(prediction[0]), 0.0)
        self.assertGreater(float(prediction[1]), 0.0)

    def test_sparse_benchmark_reports_both_models(self):
        spec = default_compressors()[0]
        telemetry = generate_compressor_telemetry(
            [spec],
            samples_per_compressor=120,
            seed=21,
            power_noise_std_kw=3.0,
        )[0]
        rows = benchmark_sparse_calibration(
            spec,
            telemetry,
            train_sizes=(8,),
            repeats=3,
            random_state=5,
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["method"] for row in rows}, {"data_only", "physics_informed"})
        self.assertTrue(all(float(row["rmse_kw"]) >= 0.0 for row in rows))

    def test_surrogate_can_be_passed_into_system_optimizer(self):
        spec = default_compressors()[0]
        telemetry = generate_compressor_telemetry(
            [spec],
            samples_per_compressor=100,
            seed=31,
            power_noise_std_kw=1.5,
        )[0]
        model = fit_physics_informed_power_surrogate(spec, telemetry)
        twin = model.to_twin(spec)
        result = optimize_system_dispatch(
            [twin],
            np.array([25.0, 30.0]),
            np.array([0.10, 0.12]),
            period_minutes=15.0,
            initial_storage_nm3=500.0,
            final_storage_min_nm3=450.0,
            storage_min_nm3=300.0,
            storage_max_nm3=700.0,
            initial_on=(0,),
            initial_flow_nm3_min=(0.0,),
        )
        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(math.isfinite(result.objective_cost))

    def test_data_only_and_physics_informed_fit_are_finite_on_small_batch(self):
        spec = default_compressors()[1]
        telemetry = generate_compressor_telemetry(
            [spec],
            samples_per_compressor=60,
            seed=44,
            power_noise_std_kw=2.5,
        )[0]
        active = np.flatnonzero(telemetry.on)[:10]
        small = TelemetryBatch(
            compressor_name=telemetry.compressor_name,
            flow_nm3_min=telemetry.flow_nm3_min[active],
            power_kw=telemetry.power_kw[active],
            on=telemetry.on[active],
        )
        for model in (
            fit_data_only_power_surrogate(spec, small),
            fit_physics_informed_power_surrogate(spec, small),
        ):
            self.assertTrue(math.isfinite(model.idle_power_kw))
            self.assertTrue(math.isfinite(model.degradation_factor))


if __name__ == "__main__":
    unittest.main()
