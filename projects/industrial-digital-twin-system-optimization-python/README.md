# Industrial Compressed-Air Digital Twin and System-Level Optimization

A machine-to-system industrial optimization project that calibrates compressor energy models from telemetry and then uses those digital-twin models inside a mixed-integer system dispatch optimizer.

The project represents a compressed-air utility plant with three compressors, an aggregate compressed-air storage/network inventory, a time-varying plant demand profile, and time-of-use electricity prices.

## Architecture

```text
compressor telemetry
(flow, power, on/off)
        |
        v
machine-level calibration
        |
        v
estimated degradation / energy models
        |
        +------------------+
        |                  |
        v                  v
compressor C1 twin     compressor C2/C3 twins
        |                  |
        +---------+--------+
                  |
                  v
         system-level MILP
                  |
                  v
 compressor commitment + flow
 storage/network inventory
 startup decisions
 shortage / vent variables
                  |
                  v
       hidden true-physics replay
```

The optimizer never receives the hidden degradation factors used by the synthetic plant simulator.

## Why compressed air

Compressed-air systems are suitable for system-level optimization because machine efficiency, compressor loading, storage, demand and electricity tariffs interact.

Optimizing each compressor independently can be inferior to optimizing the complete utility system over time.

## Machine-level model

Each compressor has:

- minimum stable flow;
- maximum flow;
- maximum ramp rate;
- idle/on-state electrical power;
- nominal marginal energy coefficient;
- hidden degradation multiplier;
- startup cost.

The hidden physical power relation is:

```text
power
=
idle power
+
nominal marginal power
* degradation factor
* flow
```

The default hidden degradation factors are:

```text
C1 = 1.04
C2 = 1.10
C3 = 1.18
```

Higher degradation means greater electrical energy consumption for the same compressed-air production.

## Synthetic telemetry

The development fixture generates noisy observations of:

```text
compressor on/off
air flow [Nm3/min]
electrical power [kW]
```

This telemetry is synthetic and is never presented as real plant sensor data.

The point of the fixture is to test whether the digital-twin calibration can recover machine efficiency degradation before the calibrated model is placed inside an optimizer.

## Calibration and validation

For each compressor, on-state telemetry is split deterministically into:

```text
75% calibration
25% held-out validation
```

An ordinary least-squares model estimates:

```text
power = intercept + slope * flow
```

The design-data marginal coefficient is treated as known, so the degradation estimate is:

```text
estimated degradation
=
estimated slope
/
nominal marginal coefficient
```

The reported RMSE is computed on held-out telemetry, not on the observations used to fit the model.

With the default seed and 2 kW telemetry noise, the development run produced:

```text
C1
estimated degradation 1.0551
held-out RMSE          2.205 kW
slope error           +1.45%

C2
estimated degradation 1.1028
held-out RMSE          2.123 kW
slope error           +0.26%

C3
estimated degradation 1.1902
held-out RMSE          1.618 kW
slope error           +0.87%
```

These are results for the synthetic fixture, not compressor measurement claims.

## System-level decision model

For compressor `i` and time period `t`, the MILP decides:

```text
q[i,t]  air flow
y[i,t]  on/off state
z[i,t]  startup indicator
s[t]    compressed-air storage/network inventory
```

It also contains nonnegative shortage and vent variables.

The balance equation is:

```text
s[t+1]
=
s[t]
+ period_minutes * (total compressor flow - demand)
+ shortage
- vent
```

Storage is expressed as equivalent standard cubic metres of air. It should be interpreted as aggregate receiver/network inventory, not as the literal geometric volume of one pressure vessel.

## Constraints

The dispatch model enforces:

- compressor minimum stable flow when on;
- compressor maximum capacity;
- binary on/off decisions;
- startup accounting;
- flow ramp limits;
- storage lower and upper bounds;
- dynamic storage balance;
- final storage requirement.

## Objective

The model minimizes:

```text
electricity cost
+ compressor startup cost
+ shortage penalty
+ vent penalty
```

Power consumption is calculated from the calibrated machine-level digital twins.

The default 24-period problem uses a representative plant-demand profile and a time-of-use electricity-price profile.

## Independent true-physics replay

After optimization, the dispatch is replayed against the hidden compressor physics.

This creates a separation between:

```text
estimated digital-twin objective
```

and

```text
realized hidden-model objective
```

It checks whether small machine-model calibration errors materially distort the system decision.

Default development run:

```text
MILP status                   OPTIMAL
estimated objective           $655.14
true-physics replay cost      $654.06
true energy                   4,566.6 kWh
shortage                      0 Nm3
vent                          0 Nm3
final storage                 700.0 Nm3
```

The estimated and hidden-model replay costs are close for this fixture.

## Rule-based baseline

A deterministic merit-order controller is included as a baseline.

It:

1. ranks compressors by estimated marginal energy intensity;
2. greedily loads the most efficient available units;
3. respects machine ramp and stable-flow limits;
4. nudges storage toward the same 700 Nm3 final-region target.

Default development run:

```text
true-physics replay cost      $722.29
true energy                   4,640.6 kWh
shortage                      0 Nm3
vent                          134.4 Nm3
final storage                 766.6 Nm3
```

Modeled replay-cost difference:

```text
$68.23
```

This is a consequence of the declared synthetic plant, tariff, demand, storage and baseline rule. It is not a real-world savings claim.

## Exact small-instance oracle

The system MILP is independently checked on a tiny two-compressor/two-period instance.

The regression oracle enumerates every binary compressor on/off schedule:

```text
2^(2 compressors * 2 periods) = 16 schedules
```

For each fixed schedule, it solves the remaining continuous dispatch problem and returns the best feasible objective.

The mixed-integer model must match this exhaustive binary oracle to numerical tolerance.

This provides an exact cross-check of the commitment/dispatch formulation on a tractable instance.

## Post-solve feasibility audit

A separate checker validates the returned dispatch without trusting the optimizer internals.

It checks:

- flow = 0 when a compressor is off;
- min/max flow when on;
- ramp constraints;
- storage limits;
- final-storage requirement;
- nonnegative shortage and vent variables.

The true-physics replay also independently reconstructs the complete storage trajectory and verifies material-balance closure.

## Regression tests

The suite covers:

- hand-checkable compressor physics;
- held-out degradation calibration;
- MILP versus exhaustive binary oracle;
- independent dispatch-feasibility audit;
- storage-balance replay;
- baseline feasibility;
- demand/tariff profile integrity.

## Run

Full 24-period experiment:

```bash
python industrial_compressed_air_digital_twin.py
```

Self-test:

```bash
python industrial_compressed_air_digital_twin.py --self-test
```

Regression tests:

```bash
python -m unittest discover -s tests -v
```

Shorter smoke run:

```bash
python industrial_compressed_air_digital_twin.py \
  --seed 42 \
  --periods 8
```

## Validated GitHub Actions run

The complete workflow was executed on GitHub Actions with Python 3.12.14. Dependency installation, the digital-twin self-test, all seven regression tests, and the full 24-period machine-to-system optimization completed successfully.

GitHub-runner result:

```text
C1 degradation estimate       1.0551
C1 held-out RMSE              2.205 kW
C2 degradation estimate       1.1028
C2 held-out RMSE              2.123 kW
C3 degradation estimate       1.1902
C3 held-out RMSE              1.618 kW

MILP status                   OPTIMAL
estimated objective           $655.14
true-physics replay cost      $654.06
true energy                   4,566.6 kWh
shortage                      0 Nm3
vent                          0 Nm3
final storage                 700.0 Nm3

rule baseline replay cost     $722.29
rule baseline energy          4,640.6 kWh
rule baseline vent            134.4 Nm3
modeled replay difference     $68.23
```

These are reproducible results for the declared synthetic fixture and seed. They are not field measurements or a real-plant savings estimate.

## Exactness versus digital-twin uncertainty

The SciPy/HiGHS MILP is exact for the declared calibrated mixed-integer dispatch problem when it returns `OPTIMAL`.

That does **not** mean the model is an exact representation of a real compressed-air plant.

The compressor telemetry, degradation states, demand and tariff are synthetic development fixtures.

A production digital twin would require, at minimum:

- calibrated compressor performance maps;
- pressure-dependent compressor and network behavior;
- receiver/network pressure measurements;
- leakage estimation;
- sensor validation and bad-data handling;
- compressor temperature and safety limits;
- maintenance/outage states;
- dynamic demand forecasting;
- model-drift monitoring;
- operational validation against plant historians.

The present model is intentionally a transparent machine-to-system optimization example rather than a plant-certified digital twin.
