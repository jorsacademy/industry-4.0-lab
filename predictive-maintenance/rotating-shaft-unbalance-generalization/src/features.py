from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

SENSORS = ("Vibration_1", "Vibration_2", "Vibration_3")


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate_hz: float = 4096.0
    order_min: float = 0.5
    order_max: float = 10.0
    order_step: float = 0.25

    @property
    def orders(self) -> np.ndarray:
        count = int(round((self.order_max - self.order_min) / self.order_step)) + 1
        return self.order_min + np.arange(count, dtype=float) * self.order_step


def _safe_float(value: float | np.floating) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _spectrum(signal: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    window = np.hanning(len(signal))
    transformed = np.fft.rfft(signal * window)
    amplitude = (2.0 / max(window.sum(), 1.0)) * np.abs(transformed)
    power = np.abs(transformed) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate_hz)
    return freqs, amplitude, power


def _interp(freqs: np.ndarray, values: np.ndarray, frequency: float) -> float:
    if frequency <= freqs[0] or frequency >= freqs[-1]:
        return 0.0
    return _safe_float(np.interp(frequency, freqs, values))


def _spectral_entropy(power: np.ndarray) -> float:
    power = np.asarray(power, dtype=float)
    total = power.sum()
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return _safe_float(-(p * np.log(p)).sum() / np.log(len(power)))


def _compact_sensor_features(signal: np.ndarray, rpm: float, cfg: FeatureConfig) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0)
    centered = x - x.mean()
    std = float(centered.std(ddof=1)) if len(centered) > 1 else 0.0
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(np.abs(centered))) if len(centered) else 0.0
    crest = peak / rms if rms > 0 else 0.0
    freqs, amplitude, power = _spectrum(centered, cfg.sample_rate_hz)
    total_power = float(power[1:].sum()) if len(power) > 1 else 0.0
    centroid = float((freqs[1:] * power[1:]).sum() / total_power) if total_power > 0 else 0.0
    shaft_hz = max(float(rpm) / 60.0, 0.0)
    amp1 = _interp(freqs, amplitude, shaft_hz)
    amp2 = _interp(freqs, amplitude, 2.0 * shaft_hz)
    amp3 = _interp(freqs, amplitude, 3.0 * shaft_hz)
    rel1 = (amp1 ** 2) / total_power if total_power > 0 else 0.0
    return {
        "rms": _safe_float(rms),
        "std": _safe_float(std),
        "p2p": _safe_float(np.ptp(x)),
        "crest": _safe_float(crest),
        "skew": _safe_float(stats.skew(x, bias=False, nan_policy="omit")),
        "kurtosis": _safe_float(stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit")),
        "spectral_entropy": _spectral_entropy(power[1:]),
        "spectral_centroid_hz": _safe_float(centroid),
        "order1_amp": amp1,
        "order2_amp": amp2,
        "order3_amp": amp3,
        "order1_rel_power": _safe_float(rel1),
        "log_broadband_power": _safe_float(np.log1p(total_power)),
    }


def extract_window_features(
    window: np.ndarray,
    columns: list[str],
    cfg: FeatureConfig,
) -> dict[str, float]:
    if window.ndim != 2:
        raise ValueError("window must be 2-D")
    index = {name: i for i, name in enumerate(columns)}
    required = {"V_in", "Measured_RPM", *SENSORS}
    missing = required.difference(index)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    rpm_values = np.asarray(window[:, index["Measured_RPM"]], dtype=float)
    vin_values = np.asarray(window[:, index["V_in"]], dtype=float)
    rpm = float(np.nanmedian(rpm_values))
    result: dict[str, float] = {
        "compact__rpm_median": _safe_float(rpm),
        "compact__rpm_mean": _safe_float(np.nanmean(rpm_values)),
        "compact__rpm_std": _safe_float(np.nanstd(rpm_values, ddof=1)),
        "compact__vin_mean": _safe_float(np.nanmean(vin_values)),
        "compact__vin_std": _safe_float(np.nanstd(vin_values, ddof=1)),
        "order__rpm_median": _safe_float(rpm),
        "order__vin_mean": _safe_float(np.nanmean(vin_values)),
    }

    shaft_hz = max(rpm / 60.0, 1e-6)
    for sensor_no, sensor in enumerate(SENSORS, start=1):
        signal = np.asarray(window[:, index[sensor]], dtype=float)
        compact = _compact_sensor_features(signal, rpm, cfg)
        for key, value in compact.items():
            result[f"compact__s{sensor_no}__{key}"] = value

        freqs, amplitude, _ = _spectrum(signal, cfg.sample_rate_hz)
        for order in cfg.orders:
            frequency = float(order * shaft_hz)
            value = np.log1p(_interp(freqs, amplitude, frequency))
            label = f"{order:.2f}".replace(".", "p")
            result[f"order__s{sensor_no}__o{label}"] = _safe_float(value)

    return result
