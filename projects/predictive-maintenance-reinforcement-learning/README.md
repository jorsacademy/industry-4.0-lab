# Predictive Maintenance with Reinforcement Learning

A reproducible industrial engineering case study for **condition-based maintenance**. The project models a degrading production asset as a Markov decision process (MDP), compares interpretable maintenance baselines, and provides training/evaluation entry points for Deep Q-Network (DQN) and Proximal Policy Optimization (PPO).

## Industrial engineering motivation

Maintenance decisions are sequential and stochastic: operating a machine earns production value but increases degradation risk; inspecting or maintaining the asset costs money and may cause downtime; catastrophic failure is rare but expensive. Reinforcement learning is appropriate when these trade-offs evolve over time and the transition dynamics are uncertain or difficult to optimize analytically.

Typical applications include CNC machines, pumps, compressors, bearings, turbines, injection-molding equipment, conveyors, and other assets with condition-monitoring signals.

## MDP formulation

**State**

- normalized degradation level
- normalized machine age
- current production load
- latest noisy sensor reading
- time since last maintenance

**Actions**

0. `continue` — keep producing
1. `inspect` — obtain a more accurate condition estimate at a small cost
2. `maintain` — preventive maintenance; partially restores condition and incurs downtime/cost
3. `replace` — reset the asset to an as-good-as-new state at the highest planned cost

**Reward**

The reward represents period contribution margin minus operating, inspection, preventive-maintenance, replacement, downtime, and failure costs. The objective is to maximize discounted lifecycle reward while reducing unplanned failures.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── src/predictive_maintenance_rl/
│   ├── __init__.py
│   ├── environment.py
│   ├── baselines.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_environment.py
│   └── test_baselines.py
├── pyproject.toml
└── README.md
```

## Installation

Core environment and tests:

```bash
python -m pip install -e .
```

RL training support:

```bash
python -m pip install -e ".[rl]"
```

Development/test dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Run the environment

```python
from predictive_maintenance_rl.environment import PredictiveMaintenanceEnv

env = PredictiveMaintenanceEnv(seed=42)
obs, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Baseline policies

The project includes three interpretable policies that are important for industrial benchmarking:

- **Run-to-failure**: always continue operating.
- **Age-based preventive maintenance**: maintain after a fixed age threshold.
- **Condition-based maintenance**: maintain when the observed degradation signal exceeds a threshold.

Run a baseline comparison:

```bash
python -m predictive_maintenance_rl.evaluate --policy baselines --episodes 100
```

## Train RL agents

DQN is a natural fit because the maintenance action space is discrete. PPO is included as a robust policy-gradient benchmark.

```bash
python -m predictive_maintenance_rl.train --algorithm dqn --timesteps 50000
python -m predictive_maintenance_rl.train --algorithm ppo --timesteps 50000
```

Models are written to `models/` by default.

Evaluate a trained model:

```bash
python -m predictive_maintenance_rl.evaluate --policy dqn --model-path models/dqn_predictive_maintenance.zip --episodes 100
```

## KPIs

Evaluation reports:

- mean episodic reward
- mean lifecycle cost proxy
- failure rate
- preventive maintenance count
- replacement count
- inspection count
- mean episode length

These are more informative for an industrial engineer than RL reward alone because they connect the learned policy to reliability, utilization, downtime, and maintenance economics.

## Experimental roadmap

The default environment is intentionally compact enough to understand and test. Natural research extensions include:

1. Weibull or proportional-hazards failure models.
2. Multi-component systems with dependent failures.
3. Remaining-useful-life estimates from vibration/temperature time series.
4. Partially observable maintenance using recurrent policies.
5. Offline RL from historical maintenance logs.
6. Constrained/safe RL with explicit availability or failure-risk limits.
7. Multi-objective optimization for cost, availability, energy, and emissions.
8. Comparison against dynamic programming, MILP, or model-predictive maintenance.

## Reproducibility and CI

The environment accepts a seed, tests check Gymnasium API behavior and policy logic, and GitHub Actions runs the test suite on supported Python versions for every push and pull request.

## License

This repository is intended for educational, research, and portfolio use. Add a project-specific license if you plan to redistribute it as a package or production component.
