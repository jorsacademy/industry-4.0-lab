from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class CompressorSpec:
    name: str
    max_flow_nm3_min: float
    min_flow_nm3_min: float
    idle_power_kw: float
    marginal_power_kw_per_nm3_min: float
    degradation_factor: float
    startup_cost: float
    max_ramp_nm3_min: float

    def true_power_kw(self, flow: float, on: bool) -> float:
        if not on:
            return 0.0
        if flow < self.min_flow_nm3_min - 1e-12 or flow > self.max_flow_nm3_min + 1e-12:
            raise ValueError("on-state flow outside stable range")
        return self.idle_power_kw + self.marginal_power_kw_per_nm3_min * self.degradation_factor * flow


@dataclass(frozen=True)
class CompressorTwin:
    name: str
    max_flow_nm3_min: float
    min_flow_nm3_min: float
    estimated_idle_power_kw: float
    estimated_marginal_power_kw_per_nm3_min: float
    estimated_degradation_factor: float
    startup_cost: float
    max_ramp_nm3_min: float

    def estimated_power_kw(self, flow: float, on: bool) -> float:
        if not on:
            return 0.0
        return self.estimated_idle_power_kw + self.estimated_marginal_power_kw_per_nm3_min * self.estimated_degradation_factor * flow


@dataclass(frozen=True)
class TelemetryBatch:
    compressor_name: str
    flow_nm3_min: np.ndarray
    power_kw: np.ndarray
    on: np.ndarray


@dataclass(frozen=True)
class TwinCalibration:
    twin: CompressorTwin
    rmse_kw: float
    estimated_slope_kw_per_flow: float
    true_slope_kw_per_flow: float


@dataclass(frozen=True)
class DispatchResult:
    objective_cost: float
    flow: np.ndarray
    on: np.ndarray
    startup: np.ndarray
    storage: np.ndarray
    shortage: np.ndarray
    vent: np.ndarray
    estimated_power_kw: np.ndarray
    status: str


@dataclass(frozen=True)
class SimulationValidation:
    realized_total_cost: float
    realized_energy_kwh: float
    realized_shortage_nm3: float
    realized_vent_nm3: float
    max_storage_balance_error: float


@dataclass(frozen=True)
class EndToEndResult:
    calibrations: Tuple[TwinCalibration, ...]
    optimized: DispatchResult
    baseline: DispatchResult
    optimized_validation: SimulationValidation
    baseline_validation: SimulationValidation


def default_compressors() -> Tuple[CompressorSpec, ...]:
    return (
        CompressorSpec("C1", 65.0, 18.0, 32.0, 1.25, 1.04, 18.0, 28.0),
        CompressorSpec("C2", 52.0, 15.0, 27.0, 1.32, 1.10, 16.0, 24.0),
        CompressorSpec("C3", 40.0, 12.0, 21.0, 1.40, 1.18, 14.0, 20.0),
    )


def generate_compressor_telemetry(
    specs: Sequence[CompressorSpec], *, samples_per_compressor: int = 120,
    seed: int = 42, power_noise_std_kw: float = 2.0,
) -> Tuple[TelemetryBatch, ...]:
    rng = np.random.default_rng(seed)
    batches = []
    for spec in specs:
        on = rng.random(samples_per_compressor) > 0.12
        flow = np.zeros(samples_per_compressor)
        idx = np.flatnonzero(on)
        flow[idx] = rng.uniform(spec.min_flow_nm3_min, spec.max_flow_nm3_min, len(idx))
        power = np.zeros(samples_per_compressor)
        power[idx] = (
            spec.idle_power_kw
            + spec.marginal_power_kw_per_nm3_min * spec.degradation_factor * flow[idx]
            + rng.normal(0.0, power_noise_std_kw, len(idx))
        )
        power[~on] = rng.normal(0.0, 0.25, np.sum(~on))
        batches.append(TelemetryBatch(spec.name, flow, power, on.astype(bool)))
    return tuple(batches)


def calibrate_compressor_twin(spec: CompressorSpec, telemetry: TelemetryBatch) -> TwinCalibration:
    if telemetry.compressor_name != spec.name:
        raise ValueError("telemetry/spec name mismatch")
    active = np.flatnonzero(telemetry.on)
    if len(active) < 20:
        raise ValueError("insufficient on-state telemetry")
    valid_idx = active[::4]
    train_idx = np.setdiff1d(active, valid_idx)
    X = np.column_stack([np.ones(len(train_idx)), telemetry.flow_nm3_min[train_idx]])
    beta, *_ = np.linalg.lstsq(X, telemetry.power_kw[train_idx], rcond=None)
    intercept, slope = map(float, beta)
    Xv = np.column_stack([np.ones(len(valid_idx)), telemetry.flow_nm3_min[valid_idx]])
    rmse = float(np.sqrt(np.mean((Xv @ beta - telemetry.power_kw[valid_idx]) ** 2)))
    degradation = slope / spec.marginal_power_kw_per_nm3_min
    twin = CompressorTwin(
        spec.name, spec.max_flow_nm3_min, spec.min_flow_nm3_min, intercept,
        spec.marginal_power_kw_per_nm3_min, degradation, spec.startup_cost,
        spec.max_ramp_nm3_min,
    )
    return TwinCalibration(
        twin, rmse, slope,
        spec.marginal_power_kw_per_nm3_min * spec.degradation_factor,
    )


def calibrate_digital_twin(specs: Sequence[CompressorSpec], telemetry: Sequence[TelemetryBatch]) -> Tuple[TwinCalibration, ...]:
    if len(specs) != len(telemetry):
        raise ValueError("spec/telemetry length mismatch")
    return tuple(calibrate_compressor_twin(s, t) for s, t in zip(specs, telemetry))


def _profile(base: Sequence[float], periods: int) -> np.ndarray:
    if periods <= 0:
        raise ValueError("periods must be positive")
    base = np.asarray(base, dtype=float)
    if periods == len(base):
        return base.copy()
    return np.interp(np.linspace(0, 1, periods), np.linspace(0, 1, len(base)), base)


def default_demand_profile(periods: int = 24) -> np.ndarray:
    return _profile([78,74,72,70,68,70,82,96,108,116,120,118,112,106,102,110,122,128,120,108,98,90,84,80], periods)


def default_electricity_price(periods: int = 24) -> np.ndarray:
    return _profile([0.085,0.080,0.078,0.076,0.078,0.085,0.105,0.125,0.145,0.155,0.160,0.150,0.138,0.132,0.140,0.165,0.190,0.205,0.188,0.160,0.135,0.115,0.100,0.092], periods)


def optimize_system_dispatch(
    twins: Sequence[CompressorTwin], demand_nm3_min: Sequence[float],
    electricity_price_per_kwh: Sequence[float], *, period_minutes: float = 60.0,
    initial_storage_nm3: float = 900.0, final_storage_min_nm3: float | None = 700.0,
    storage_min_nm3: float = 300.0, storage_max_nm3: float = 1800.0,
    shortage_penalty_per_nm3: float = 4.0, vent_penalty_per_nm3: float = 0.05,
    initial_on: Sequence[int] | None = None,
    initial_flow_nm3_min: Sequence[float] | None = None,
) -> DispatchResult:
    twins = tuple(twins)
    demand = np.asarray(demand_nm3_min, float)
    price = np.asarray(electricity_price_per_kwh, float)
    n, T = len(twins), len(demand)
    if n == 0 or T == 0 or len(price) != T:
        raise ValueError("invalid system dimensions")
    if not storage_min_nm3 <= initial_storage_nm3 <= storage_max_nm3:
        raise ValueError("initial storage outside bounds")
    dt_h, dt_min = period_minutes / 60.0, period_minutes
    initial_on = np.zeros(n, int) if initial_on is None else np.asarray(initial_on, int)
    initial_flow = np.zeros(n) if initial_flow_nm3_min is None else np.asarray(initial_flow_nm3_min, float)

    q0, y0, z0 = 0, n*T, 2*n*T
    s0, sh0, v0 = 3*n*T, 3*n*T + T + 1, 3*n*T + 2*T + 1
    nv = 3*n*T + 3*T + 1
    qi = lambda i,t: q0 + i*T + t
    yi = lambda i,t: y0 + i*T + t
    zi = lambda i,t: z0 + i*T + t
    si = lambda t: s0 + t
    shi = lambda t: sh0 + t
    vi = lambda t: v0 + t

    c = np.zeros(nv)
    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    integrality = np.zeros(nv, int)
    for i, twin in enumerate(twins):
        slope = twin.estimated_marginal_power_kw_per_nm3_min * twin.estimated_degradation_factor
        for t in range(T):
            c[qi(i,t)] = price[t] * dt_h * slope
            c[yi(i,t)] = price[t] * dt_h * twin.estimated_idle_power_kw
            c[zi(i,t)] = twin.startup_cost
            ub[qi(i,t)] = twin.max_flow_nm3_min
            ub[yi(i,t)] = ub[zi(i,t)] = 1.0
            integrality[yi(i,t)] = integrality[zi(i,t)] = 1
    for t in range(T + 1):
        lb[si(t)], ub[si(t)] = storage_min_nm3, storage_max_nm3
    lb[si(0)] = ub[si(0)] = initial_storage_nm3
    if final_storage_min_nm3 is not None:
        lb[si(T)] = max(lb[si(T)], final_storage_min_nm3)
    for t in range(T):
        c[shi(t)], c[vi(t)] = shortage_penalty_per_nm3, vent_penalty_per_nm3

    rows, lo, hi = [], [], []
    def add(row, lower=-np.inf, upper=np.inf):
        rows.append(row); lo.append(lower); hi.append(upper)

    for i, twin in enumerate(twins):
        for t in range(T):
            r = np.zeros(nv); r[qi(i,t)] = 1; r[yi(i,t)] = -twin.max_flow_nm3_min; add(r, upper=0)
            r = np.zeros(nv); r[qi(i,t)] = 1; r[yi(i,t)] = -twin.min_flow_nm3_min; add(r, lower=0)
            r = np.zeros(nv); r[zi(i,t)] = 1; r[yi(i,t)] = -1
            if t == 0:
                add(r, lower=-float(initial_on[i]))
            else:
                r[yi(i,t-1)] = 1; add(r, lower=0)
            r = np.zeros(nv); r[qi(i,t)] = 1
            if t == 0:
                add(r, initial_flow[i] - twin.max_ramp_nm3_min, initial_flow[i] + twin.max_ramp_nm3_min)
            else:
                r[qi(i,t-1)] = -1; add(r, -twin.max_ramp_nm3_min, twin.max_ramp_nm3_min)

    for t in range(T):
        r = np.zeros(nv); r[si(t+1)] = 1; r[si(t)] = -1
        for i in range(n): r[qi(i,t)] = -dt_min
        r[shi(t)] = -1; r[vi(t)] = 1
        rhs = -dt_min * demand[t]; add(r, rhs, rhs)

    res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub),
               constraints=LinearConstraint(np.vstack(rows), np.asarray(lo), np.asarray(hi)),
               options={"time_limit": 60.0})
    if res.x is None:
        raise RuntimeError("system MILP returned no solution")
    x = res.x
    flow = np.array([[x[qi(i,t)] for t in range(T)] for i in range(n)])
    on = np.array([[x[yi(i,t)] > 0.5 for t in range(T)] for i in range(n)], bool)
    startup = np.array([[x[zi(i,t)] > 0.5 for t in range(T)] for i in range(n)], bool)
    storage = np.array([x[si(t)] for t in range(T+1)])
    shortage = np.array([x[shi(t)] for t in range(T)])
    vent = np.array([x[vi(t)] for t in range(T)])
    power = np.zeros((n,T))
    for i, twin in enumerate(twins):
        power[i] = twin.estimated_idle_power_kw * on[i] + twin.estimated_marginal_power_kw_per_nm3_min * twin.estimated_degradation_factor * flow[i]
    return DispatchResult(float(res.fun), flow, on, startup, storage, shortage, vent, power,
                          "OPTIMAL" if res.status == 0 else "FEASIBLE_LIMIT")


def max_dispatch_constraint_violation(
    twins: Sequence[CompressorTwin], dispatch: DispatchResult, *,
    initial_flow_nm3_min: Sequence[float] | None = None,
    storage_min_nm3: float = 300.0, storage_max_nm3: float = 1800.0,
    final_storage_min_nm3: float | None = None,
) -> float:
    twins = tuple(twins); n, T = dispatch.flow.shape
    initial = np.zeros(n) if initial_flow_nm3_min is None else np.asarray(initial_flow_nm3_min, float)
    violations = [0.0]
    for i, twin in enumerate(twins):
        prev = initial[i]
        for t in range(T):
            q, y = dispatch.flow[i,t], dispatch.on[i,t]
            if y:
                violations += [twin.min_flow_nm3_min - q, q - twin.max_flow_nm3_min]
            else:
                violations.append(abs(q))
            violations.append(abs(q - prev) - twin.max_ramp_nm3_min); prev = q
    violations += list(storage_min_nm3 - dispatch.storage) + list(dispatch.storage - storage_max_nm3)
    violations += list(-dispatch.shortage) + list(-dispatch.vent)
    if final_storage_min_nm3 is not None:
        violations.append(final_storage_min_nm3 - dispatch.storage[-1])
    return float(max(0.0, np.max(violations)))


def simulate_dispatch_with_true_physics(
    specs: Sequence[CompressorSpec], dispatch: DispatchResult,
    demand_nm3_min: Sequence[float], electricity_price_per_kwh: Sequence[float], *,
    period_minutes: float = 60.0, shortage_penalty_per_nm3: float = 4.0,
    vent_penalty_per_nm3: float = 0.05,
) -> SimulationValidation:
    specs = tuple(specs); demand = np.asarray(demand_nm3_min, float); price = np.asarray(electricity_price_per_kwh, float)
    n, T = dispatch.flow.shape; dt_h, dt_min = period_minutes/60.0, period_minutes
    power = np.zeros((n,T))
    for i, spec in enumerate(specs):
        for t in range(T):
            if dispatch.on[i,t]: power[i,t] = spec.true_power_kw(dispatch.flow[i,t], True)
    energy = float(np.sum(power) * dt_h)
    energy_cost = float(np.sum(power * price[None,:]) * dt_h)
    starts = sum(specs[i].startup_cost for i in range(n) for t in range(T) if dispatch.startup[i,t])
    total = energy_cost + starts + shortage_penalty_per_nm3*np.sum(dispatch.shortage) + vent_penalty_per_nm3*np.sum(dispatch.vent)
    storage = np.empty(T+1); storage[0] = dispatch.storage[0]
    for t in range(T):
        storage[t+1] = storage[t] + dt_min*(np.sum(dispatch.flow[:,t]) - demand[t]) + dispatch.shortage[t] - dispatch.vent[t]
    return SimulationValidation(float(total), energy, float(np.sum(dispatch.shortage)), float(np.sum(dispatch.vent)), float(np.max(np.abs(storage-dispatch.storage))))


def rule_based_baseline(
    twins: Sequence[CompressorTwin], demand_nm3_min: Sequence[float], electricity_price_per_kwh: Sequence[float], *,
    period_minutes: float = 60.0, initial_storage_nm3: float = 900.0, storage_target_nm3: float = 900.0,
    storage_min_nm3: float = 300.0, storage_max_nm3: float = 1800.0,
    shortage_penalty_per_nm3: float = 4.0, vent_penalty_per_nm3: float = 0.05,
) -> DispatchResult:
    twins = tuple(twins); demand = np.asarray(demand_nm3_min, float); price = np.asarray(electricity_price_per_kwh, float)
    n, T = len(twins), len(demand); dt_min, dt_h = period_minutes, period_minutes/60.0
    flow = np.zeros((n,T)); on = np.zeros((n,T), bool); startup = np.zeros((n,T), bool); power = np.zeros((n,T))
    storage = np.zeros(T+1); storage[0] = initial_storage_nm3; shortage = np.zeros(T); vent = np.zeros(T)
    prev_flow = np.zeros(n); prev_on = np.zeros(n, bool)
    order = sorted(range(n), key=lambda i: twins[i].estimated_marginal_power_kw_per_nm3_min * twins[i].estimated_degradation_factor)
    for t in range(T):
        target = max(0.0, demand[t] + 0.35*(storage_target_nm3-storage[t])/dt_min)
        remaining = target
        for i in order:
            twin = twins[i]
            max_reachable = min(twin.max_flow_nm3_min, prev_flow[i] + twin.max_ramp_nm3_min)
            if remaining <= 1e-9: continue
            q = min(max_reachable, remaining)
            if q < twin.min_flow_nm3_min: q = min(max_reachable, twin.min_flow_nm3_min)
            if q >= twin.min_flow_nm3_min - 1e-9:
                flow[i,t] = q; on[i,t] = True; remaining -= q
        next_storage = storage[t] + dt_min*(np.sum(flow[:,t])-demand[t])
        if next_storage < storage_min_nm3:
            shortage[t] = storage_min_nm3-next_storage; next_storage = storage_min_nm3
        elif next_storage > storage_max_nm3:
            vent[t] = next_storage-storage_max_nm3; next_storage = storage_max_nm3
        storage[t+1] = next_storage
        for i, twin in enumerate(twins):
            startup[i,t] = on[i,t] and not prev_on[i]
            power[i,t] = twin.estimated_power_kw(flow[i,t], bool(on[i,t]))
        prev_flow, prev_on = flow[:,t].copy(), on[:,t].copy()
    objective = float(np.sum(power*price[None,:])*dt_h + sum(twins[i].startup_cost for i in range(n) for t in range(T) if startup[i,t]) + shortage_penalty_per_nm3*np.sum(shortage) + vent_penalty_per_nm3*np.sum(vent))
    return DispatchResult(objective, flow, on, startup, storage, shortage, vent, power, "RULE_BASELINE")


def brute_force_on_off_oracle(
    twins: Sequence[CompressorTwin], demand_nm3_min: Sequence[float], electricity_price_per_kwh: Sequence[float], *,
    period_minutes: float, initial_storage_nm3: float, storage_min_nm3: float,
    storage_max_nm3: float, final_storage_min_nm3: float,
) -> float:
    twins = tuple(twins); demand = np.asarray(demand_nm3_min,float); price = np.asarray(electricity_price_per_kwh,float)
    n, T = len(twins), len(demand)
    if n*T > 12: raise ValueError("oracle instance too large")
    dt_h, dt_min = period_minutes/60.0, period_minutes; best = math.inf
    for bits in itertools.product((0,1), repeat=n*T):
        y = np.array(bits,int).reshape(n,T)
        qn, sn, shn, vn = n*T, T+1, T, T; nv = qn+sn+shn+vn
        qi=lambda i,t:i*T+t; si=lambda t:qn+t; shi=lambda t:qn+sn+t; vi=lambda t:qn+sn+shn+t
        c=np.zeros(nv); lb=np.zeros(nv); ub=np.full(nv,np.inf); fixed=0.0
        for i,twin in enumerate(twins):
            slope=twin.estimated_marginal_power_kw_per_nm3_min*twin.estimated_degradation_factor
            for t in range(T):
                c[qi(i,t)] = price[t]*dt_h*slope
                if y[i,t]:
                    lb[qi(i,t)],ub[qi(i,t)] = twin.min_flow_nm3_min,twin.max_flow_nm3_min
                    fixed += price[t]*dt_h*twin.estimated_idle_power_kw
                    if t==0 or not y[i,t-1]: fixed += twin.startup_cost
                else: ub[qi(i,t)] = 0.0
        for t in range(T+1): lb[si(t)],ub[si(t)] = storage_min_nm3,storage_max_nm3
        lb[si(0)] = ub[si(0)] = initial_storage_nm3; lb[si(T)] = max(lb[si(T)],final_storage_min_nm3)
        for t in range(T): c[shi(t)],c[vi(t)] = 4.0,0.05
        rows=[]; lo=[]; hi=[]
        for i,twin in enumerate(twins):
            for t in range(T):
                r=np.zeros(nv); r[qi(i,t)]=1
                if t==0: rows.append(r);lo.append(-twin.max_ramp_nm3_min);hi.append(twin.max_ramp_nm3_min)
                else: r[qi(i,t-1)]=-1;rows.append(r);lo.append(-twin.max_ramp_nm3_min);hi.append(twin.max_ramp_nm3_min)
        for t in range(T):
            r=np.zeros(nv);r[si(t+1)]=1;r[si(t)]=-1
            for i in range(n):r[qi(i,t)]=-dt_min
            r[shi(t)]=-1;r[vi(t)]=1;rhs=-dt_min*demand[t];rows.append(r);lo.append(rhs);hi.append(rhs)
        res=milp(c=c,integrality=np.zeros(nv,int),bounds=Bounds(lb,ub),constraints=LinearConstraint(np.vstack(rows),np.asarray(lo),np.asarray(hi)))
        if res.x is not None: best=min(best,fixed+float(res.fun))
    if not math.isfinite(best): raise RuntimeError("oracle found no feasible schedule")
    return best


def run_end_to_end(*, seed: int = 42, periods: int = 24) -> EndToEndResult:
    specs = default_compressors()
    telemetry = generate_compressor_telemetry(specs, seed=seed)
    calibrations = calibrate_digital_twin(specs, telemetry)
    twins = tuple(c.twin for c in calibrations)
    demand, price = default_demand_profile(periods), default_electricity_price(periods)
    optimized = optimize_system_dispatch(twins, demand, price)
    baseline = rule_based_baseline(twins, demand, price, storage_target_nm3=700.0)
    opt_val = simulate_dispatch_with_true_physics(specs, optimized, demand, price)
    base_val = simulate_dispatch_with_true_physics(specs, baseline, demand, price)
    if optimized.status != "OPTIMAL": raise RuntimeError("default MILP did not prove optimality")
    if max_dispatch_constraint_violation(twins, optimized, final_storage_min_nm3=700.0) > 1e-7: raise RuntimeError("optimized dispatch infeasible")
    if max_dispatch_constraint_violation(twins, baseline) > 1e-7: raise RuntimeError("baseline dispatch infeasible")
    if max(opt_val.max_storage_balance_error, base_val.max_storage_balance_error) > 1e-7: raise RuntimeError("storage replay mismatch")
    return EndToEndResult(calibrations, optimized, baseline, opt_val, base_val)


def self_test() -> None:
    specs = default_compressors()
    telemetry = generate_compressor_telemetry(specs, samples_per_compressor=300, seed=7, power_noise_std_kw=0.8)
    calibrations = calibrate_digital_twin(specs, telemetry)
    for cal in calibrations:
        assert abs(cal.estimated_slope_kw_per_flow-cal.true_slope_kw_per_flow)/cal.true_slope_kw_per_flow < 0.03
        assert cal.rmse_kw < 1.2
    spec=specs[0]; expected=spec.idle_power_kw+spec.marginal_power_kw_per_nm3_min*spec.degradation_factor*30.0
    assert math.isclose(spec.true_power_kw(30.0,True),expected,abs_tol=1e-12)
    twins=tuple(c.twin for c in calibrations); demand=np.array([48.0,64.0]); price=np.array([0.10,0.20])
    result=optimize_system_dispatch(twins[:2],demand,price,period_minutes=15.0,initial_storage_nm3=500.0,final_storage_min_nm3=450.0,storage_min_nm3=300.0,storage_max_nm3=700.0,initial_on=(0,0),initial_flow_nm3_min=(0.0,0.0))
    oracle=brute_force_on_off_oracle(twins[:2],demand,price,period_minutes=15.0,initial_storage_nm3=500.0,storage_min_nm3=300.0,storage_max_nm3=700.0,final_storage_min_nm3=450.0)
    assert result.status=="OPTIMAL" and math.isclose(result.objective_cost,oracle,abs_tol=1e-7)
    assert max_dispatch_constraint_violation(twins[:2],result,storage_min_nm3=300.0,storage_max_nm3=700.0,final_storage_min_nm3=450.0) < 1e-8
    assert simulate_dispatch_with_true_physics(specs[:2],result,demand,price,period_minutes=15.0).max_storage_balance_error < 1e-8
    print("Industrial compressed-air digital twin self-test: OK")


def print_result(result: EndToEndResult) -> None:
    print("="*88); print("INDUSTRIAL COMPRESSED-AIR DIGITAL TWIN + SYSTEM-LEVEL OPTIMIZATION"); print("="*88)
    print("Machine-level twin calibration")
    for cal in result.calibrations:
        err=100*(cal.estimated_slope_kw_per_flow-cal.true_slope_kw_per_flow)/cal.true_slope_kw_per_flow
        print(f"  {cal.twin.name}: degradation={cal.twin.estimated_degradation_factor:.4f} RMSE={cal.rmse_kw:.3f} kW slope error={err:+.2f}%")
    opt,base=result.optimized,result.baseline;ov,bv=result.optimized_validation,result.baseline_validation
    print("\nSystem-level MILP")
    print(f"  status                       : {opt.status}\n  estimated objective          : ${opt.objective_cost:,.2f}\n  true-physics replay cost     : ${ov.realized_total_cost:,.2f}\n  true energy                  : {ov.realized_energy_kwh:,.1f} kWh\n  total shortage               : {ov.realized_shortage_nm3:,.3f} Nm3\n  total vent                   : {ov.realized_vent_nm3:,.3f} Nm3\n  final storage                : {opt.storage[-1]:,.1f} Nm3")
    print("\nRule-based baseline")
    print(f"  estimated objective          : ${base.objective_cost:,.2f}\n  true-physics replay cost     : ${bv.realized_total_cost:,.2f}\n  true energy                  : {bv.realized_energy_kwh:,.1f} kWh\n  total shortage               : {bv.realized_shortage_nm3:,.3f} Nm3\n  total vent                   : {bv.realized_vent_nm3:,.3f} Nm3\n  final storage                : {base.storage[-1]:,.1f} Nm3")
    print(f"\nModeled replay cost difference : ${bv.realized_total_cost-ov.realized_total_cost:,.2f}")
    print("The MILP is exact for the declared calibrated dispatch model. The digital twin is estimated from synthetic telemetry and is not a calibrated real-plant model.")


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true");parser.add_argument("--seed",type=int,default=42);parser.add_argument("--periods",type=int,default=24);args=parser.parse_args()
    if args.self_test:self_test()
    else:print_result(run_end_to_end(seed=args.seed,periods=args.periods))


if __name__ == "__main__":
    main()
