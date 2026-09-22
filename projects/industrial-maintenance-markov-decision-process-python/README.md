# Industrial Maintenance Markov Decision Process

A fully observed condition-based maintenance optimization example using Markov chains and an infinite-horizon discounted Markov Decision Process (MDP).

The asset condition is reviewed at regular discrete intervals:

```text
Healthy -> Degraded -> Critical -> Failed
```

At each review, the decision maker chooses one of:

```text
Operate
Minor maintenance
Replace
```

Failed equipment must be replaced.

## Objective

The model minimizes expected infinite-horizon discounted cost:

```text
E[ sum gamma^t * cost(S_t, A_t) ]
```

with default discount factor `gamma = 0.97`.

Costs are in stylized `$1,000` units per review period. They represent educational engineering assumptions, not calibrated plant accounting data.

## Operating degradation model

Default `Operate` transition matrix:

```text
             Next state
Current      Healthy  Degraded Critical Failed

Healthy       0.880    0.105    0.014   0.001
Degraded      0.000    0.730    0.220   0.050
Critical      0.000    0.000    0.520   0.480
Failed        0.000    0.000    0.000   1.000
```

Minor maintenance is imperfect and probabilistically improves condition. Replacement returns the asset to `Healthy` with probability `0.995` and `Degraded` with probability `0.005`.

Default state costs:

```text
Healthy       0
Degraded     35
Critical    160
Failed      850
```

Default action costs:

```text
Operate              0
Minor maintenance  120
Replace            420
```

## Independent exact solution methods

The same finite discounted MDP is solved with three independent methods:

1. value iteration;
2. policy iteration;
3. exhaustive enumeration of every valid stationary deterministic policy.

There are exactly `3 * 3 * 3 * 1 = 27` valid stationary deterministic policies in the default model.

For a fixed policy, the code solves:

```text
(I - gamma P_pi) V_pi = c_pi
```

exactly up to floating-point numerical precision.

## Validated default policy

All three exact methods return:

```text
Healthy   -> Operate
Degraded  -> MinorMaintenance
Critical  -> Replace
Failed    -> Replace
```

Default discounted values:

```text
Healthy      967.309
Degraded    1166.810
Critical    1519.257
Failed      2209.257
```

Bellman residual:

```text
2.274e-13
```

## Monte Carlo cross-check

Monte Carlo simulation is not used to optimize the policy. It independently checks the exact policy value.

The GitHub Actions smoke configuration uses 3,000 trajectories from each initial state and a 220-period horizon.

## Markov-chain diagnostics

Once a stationary policy is fixed, it induces a four-state Markov chain. The repository computes its stationary distribution, long-run average one-period cost and stationary failure probability.

Default optimal-policy diagnostics:

```text
V(Healthy)             967.309
stationary avg cost     30.077
stationary P(Failed)     0.111%
```

Baseline diagnostics:

```text
Run-to-failure
V(Healthy)            3458.233
stationary avg cost    116.387
stationary P(Failed)     7.011%

Critical-state replacement
V(Healthy)            1976.287
stationary avg cost     64.709
stationary P(Failed)     1.360%
```

These are consequences of the declared model, not measured industrial failure rates.

## Sensitivity analysis

The optimal Critical-state decision changes when failure consequences change:

```text
Failed-state cost 400 -> Critical: MinorMaintenance
Failed-state cost 850 -> Critical: Replace
```

This demonstrates that the maintenance policy depends on reliability and economic assumptions rather than on one universal threshold.

## Validation strategy

Regression tests verify transition probabilities, action restrictions, a hand-checkable Bellman backup, agreement of all three exact methods, exhaustive enumeration of all 27 policies, dominance over two baselines, stationary-distribution invariance, Monte Carlo reproducibility, and failure-cost sensitivity.

## Validated GitHub Actions run

The full CI workflow was executed with Python 3.12.14. The self-test, all 9 regression tests and the stochastic cross-check completed successfully.

GitHub-runner stochastic check:

```text
Value-iteration iterations    947
Policy-iteration iterations     2
Policies exhaustively checked  27
Bellman residual             2.274e-13

Healthy   MC  966.462   exact  967.309
Degraded  MC 1164.765   exact 1166.810
Critical  MC 1521.349   exact 1519.257
Failed    MC 2207.165   exact 2209.257
```

The Monte Carlo numbers are stochastic estimates; the dynamic-programming values are the exact numerical solution for the declared finite MDP.

## Run

```bash
python industrial_maintenance_mdp.py
```

Self-test:

```bash
python industrial_maintenance_mdp.py --self-test
```

Regression suite:

```bash
python -m unittest discover -s tests -v
```

CI-style stochastic run:

```bash
python industrial_maintenance_mdp.py \
  --monte-carlo-trajectories 3000 \
  --monte-carlo-horizon 220 \
  --seed 42
```

## Exactness versus modeling scope

The dynamic-programming solution is exact for the declared finite discounted MDP up to numerical tolerance.

That does **not** imply that the resulting policy is optimal for a real plant. Production use would require health-state definitions, transition probabilities, maintenance effectiveness and economic costs to be estimated and validated from actual operational data.

The model is fully observed. If equipment health were latent and only noisy sensor observations were available, a partially observable MDP (POMDP) would be a more appropriate formulation.
