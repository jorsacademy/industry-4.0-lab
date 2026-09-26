from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import lsq_linear

from industrial_compressed_air_digital_twin import (
    CompressorSpec,
    CompressorTwin,
    TelemetryBatch,
    default_compressors,
    generate_compressor_telemetry,
)


@dataclass(frozen=True)
class PowerSurrogate:
    idle_power_kw: float
    nominal_marginal_power_kw_per_nm3_min: float
    degradation_factor: float
    method: str

    def predict(self, flow_nm3_min: np.ndarray | float, on: np.ndarray | bool) -> np.ndarray:
        flow = np.asarray(flow_nm3_min, dtype=float)
        on_state = np.asarray(on, dtype=bool)
        active_power = (
            self.idle_power_kw
            + self.nominal_marginal_power_kw_per_nm3_min
            * self.degradation_factor
            * flow
        )
        return np.where(on_state, active_power, 0.0)

    def to_twin(self, spec: CompressorSpec) -> CompressorTwin:
        return CompressorTwin(
            name=spec.name,
            max_flow_nm3_min=spec.max_flow_nm3_min,
            min_flow_nm3_min=spec.min_flow_nm3_min,
            estimated_idle_power_kw=self.idle_power_kw,
            estimated_marginal_power_kw_per_nm3_min=spec.marginal_power_kw_per_nm3_min,
            estimated_degradation_factor=self.degradation_factor,
            startup_cost=spec.startup_cost,
            max_ramp_nm3_min=spec.max_ramp_nm3_min,
        )


def _active_arrays(telemetry: TelemetryBatch) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(telemetry.on)
    if len(active) < 3:
        raise ValueError("at least three on-state samples are required")
    return (
        telemetry.flow_nm3_min[active].astype(float),
        telemetry.power_kw[active].astype(float),
    )


def fit_data_only_power_surrogate(
    spec: CompressorSpec,
    telemetry: TelemetryBatch,
) -> PowerSurrogate:
    flow, power = _active_arrays(telemetry)
    design = np.column_stack([np.ones(len(flow)), flow])
    beta, *_ = np.linalg.lstsq(design, power, rcond=None)
    idle = max(0.0, float(beta[0]))
    slope = max(0.0, float(beta[1]))
    degradation = slope / spec.marginal_power_kw_per_nm3_min
    return PowerSurrogate(
        idle_power_kw=idle,
        nominal_marginal_power_kw_per_nm3_min=spec.marginal_power_kw_per_nm3_min,
        degradation_factor=degradation,
        method="data_only",
    )


def fit_physics_informed_power_surrogate(
    spec: CompressorSpec,
    telemetry: TelemetryBatch,
    *,
    idle_prior_weight: float = 6.0,
    degradation_prior_weight: float = 6.0,
    max_degradation_factor: float = 3.0,
) -> PowerSurrogate:
    if idle_prior_weight < 0 or degradation_prior_weight < 0:
        raise ValueError("prior weights must be non-negative")
    if max_degradation_factor <= 0:
        raise ValueError("max_degradation_factor must be positive")

    flow, power = _active_arrays(telemetry)

    design = np.column_stack(
        [
            np.ones(len(flow)),
            spec.marginal_power_kw_per_nm3_min * flow,
        ]
    )
    rows = [design]
    targets = [power]

    if idle_prior_weight > 0:
        rows.append(
            np.array(
                [[np.sqrt(idle_prior_weight), 0.0]],
                dtype=float,
            )
        )
        targets.append(
            np.array(
                [np.sqrt(idle_prior_weight) * spec.idle_power_kw],
                dtype=float,
            )
        )

    if degradation_prior_weight > 0:
        rows.append(
            np.array(
                [[0.0, np.sqrt(degradation_prior_weight)]],
                dtype=float,
            )
        )
        targets.append(
            np.array(
                [np.sqrt(degradation_prior_weight)],
                dtype=float,
            )
        )

    augmented_x = np.vstack(rows)
    augmented_y = np.concatenate(targets)
    result = lsq_linear(
        augmented_x,
        augmented_y,
        bounds=(
            np.array([0.0, 0.0]),
            np.array([np.inf, max_degradation_factor]),
        ),
    )
    if not result.success:
        raise RuntimeError(f"physics-informed calibration failed: {result.message}")

    return PowerSurrogate(
        idle_power_kw=float(result.x[0]),
        nominal_marginal_power_kw_per_nm3_min=spec.marginal_power_kw_per_nm3_min,
        degradation_factor=float(result.x[1]),
        method="physics_informed",
    )


def surrogate_rmse(
    surrogate: PowerSurrogate,
    telemetry: TelemetryBatch,
) -> float:
    prediction = surrogate.predict(
        telemetry.flow_nm3_min,
        telemetry.on,
    )
    return float(np.sqrt(np.mean((prediction - telemetry.power_kw) ** 2)))


def _subset_telemetry(
    telemetry: TelemetryBatch,
    indices: np.ndarray,
) -> TelemetryBatch:
    indices = np.asarray(indices, dtype=int)
    return TelemetryBatch(
        compressor_name=telemetry.compressor_name,
        flow_nm3_min=telemetry.flow_nm3_min[indices],
        power_kw=telemetry.power_kw[indices],
        on=telemetry.on[indices],
    )


def benchmark_sparse_calibration(
    spec: CompressorSpec,
    telemetry: TelemetryBatch,
    *,
    train_sizes: Iterable[int] = (8, 16, 32, 64),
    repeats: int = 20,
    random_state: int = 42,
    idle_prior_weight: float = 6.0,
    degradation_prior_weight: float = 6.0,
) -> list[dict[str, float | int | str]]:
    active = np.flatnonzero(telemetry.on)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []

    for train_size in map(int, train_sizes):
        if train_size < 3 or train_size >= len(active):
            raise ValueError("each train size must be at least 3 and smaller than the number of active samples")

        for repeat in range(repeats):
            train_idx = np.asarray(
                rng.choice(active, size=train_size, replace=False),
                dtype=int,
            )
            test_idx = np.setdiff1d(active, train_idx, assume_unique=False)
            train = _subset_telemetry(telemetry, train_idx)
            test = _subset_telemetry(telemetry, test_idx)

            data_only = fit_data_only_power_surrogate(spec, train)
            physics_informed = fit_physics_informed_power_surrogate(
                spec,
                train,
                idle_prior_weight=idle_prior_weight,
                degradation_prior_weight=degradation_prior_weight,
            )

            for surrogate in (data_only, physics_informed):
                rows.append(
                    {
                        "compressor": spec.name,
                        "method": surrogate.method,
                        "train_size": train_size,
                        "repeat": repeat,
                        "rmse_kw": surrogate_rmse(surrogate, test),
                        "idle_power_kw": surrogate.idle_power_kw,
                        "degradation_factor": surrogate.degradation_factor,
                    }
                )

    return rows


def _summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    keys = sorted(
        {
            (str(row["compressor"]), str(row["method"]), int(row["train_size"]))
            for row in rows
        }
    )
    summary: list[dict[str, float | int | str]] = []
    for compressor, method, train_size in keys:
        values = np.array(
            [
                float(row["rmse_kw"])
                for row in rows
                if row["compressor"] == compressor
                and row["method"] == method
                and int(row["train_size"]) == train_size
            ],
            dtype=float,
        )
        summary.append(
            {
                "compressor": compressor,
                "method": method,
                "train_size": train_size,
                "mean_rmse_kw": float(values.mean()),
                "median_rmse_kw": float(np.median(values)),
                "std_rmse_kw": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--noise-kw", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    specs = default_compressors()
    telemetry = generate_compressor_telemetry(
        specs,
        samples_per_compressor=args.samples,
        seed=args.seed,
        power_noise_std_kw=args.noise_kw,
    )

    rows: list[dict[str, float | int | str]] = []
    for index, (spec, batch) in enumerate(zip(specs, telemetry)):
        rows.extend(
            benchmark_sparse_calibration(
                spec,
                batch,
                repeats=args.repeats,
                random_state=args.seed + index,
            )
        )

    print(
        json.dumps(
            {
                "experiment": "sparse-data physics-informed compressor calibration",
                "note": (
                    "The physics-informed model uses the known compressor form, OEM nominal slope, "
                    "non-negative parameters, zero off-state power, and soft priors. It is not a PINN."
                ),
                "summary": _summarize(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
