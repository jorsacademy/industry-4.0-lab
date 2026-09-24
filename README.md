# Industry 4.0 Lab

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`dynamic-manufacturing-digital-twin-rl`](projects/dynamic-manufacturing-digital-twin-rl/)
- [`industrial-digital-twin-system-optimization-python`](projects/industrial-digital-twin-system-optimization-python/)
- [`industrial-maintenance-markov-decision-process-python`](projects/industrial-maintenance-markov-decision-process-python/)
- [`industrial-maintenance-scheduling-optimizer`](projects/industrial-maintenance-scheduling-optimizer/)
- [`predictive-maintenance-reinforcement-learning`](projects/predictive-maintenance-reinforcement-learning/)
- [`predictive-maintenance-sensor-data-optimization-python`](projects/predictive-maintenance-sensor-data-optimization-python/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

A curated portfolio of applied Industry 4.0 projects built around cyber-physical production systems, industrial sensing, IIoT, PLC/SCADA data, condition monitoring, predictive maintenance, process analytics, digital twins, and optimization.

## Project map

| # | Domain | Project | Industrial data | Main objective | Status |
|---|---|---|---|---|---|
| 01 | Predictive Maintenance | [CNC Tool Wear & Process Health Monitoring](predictive-maintenance/cnc-tool-wear/) | 10 Hz multi-axis CNC servo and spindle telemetry | Detect tool wear and support online condition monitoring | Ready |
| 02 | Smart Quality | [Electrical Test Report — Smart Quality Monitoring](quality-monitoring/electrical-test-report/) | End-of-line electrical test measurements, lot/time traceability, `F1..F19` | Defect risk scoring, lot monitoring, and adaptive inspection | Ready |
| 03 | Smart Quality | [Bosch Production Line — Early Defect Risk & Smart Quality](quality-monitoring/bosch-production-line/) | Station-tagged production measurements and measurement times across >1M parts | Rare-defect prediction, early warning, active inspection, and few-label adaptation | Ready |
| 04 | Semiconductor Process Monitoring | [SECOM Semiconductor Yield — Process Monitoring & Sensor-Panel Rationalization](quality-monitoring/secom-semiconductor-yield/) | 590 anonymous real process/sensor measurements with timestamped pass/fail yield | Rare-failure detection, compact monitoring-panel selection, and drift diagnostics | Ready |
| 05 | Process Optimization | [CNC Turning — Surface Integrity, Cutting Forces & Process Trade-offs](process-optimization/cnc-turning-quality-force/) | Real turning DoE with dynamometer forces, roughness probes and flank-wear levels | Run-safe response modeling, sensor-value analysis, and Pareto process trade-offs | Ready |
| 06 | Condition Monitoring | [Multimodal Motor Condition Monitoring — Acoustic + Vibration Diagnosis](predictive-maintenance/multimodal-motor-condition-monitoring/) | Measured 3-axis structure vibration + microphone frequency features across 8 rig conditions | Leakage-aware blocked diagnosis and sensor-modality ablation | Ready |
| 07 | Metallurgical Process Monitoring | [Electric Arc Furnace — Terminal Temperature Soft Sensor](process-monitoring/electric-arc-furnace-soft-sensor/) | Heat-level EAF temperature/oxidation, transformer, gas/oxygen, carbon-injection and material-event logs | Leakage-safe terminal-temperature forecasting with chronological uncertainty calibration | Ready |
| 08 | Rotating Machinery | [Rotating-Shaft Unbalance — Independent-Session Generalization](predictive-maintenance/rotating-shaft-unbalance-generalization/) | 4096 Hz three-sensor vibration with measured RPM across separate D/E sessions and five unbalance severities | Cross-session diagnosis, calibration, RPM diagnostics, and CORAL/few-shot adaptation | Ready |
| 09 | Continuous Process Monitoring | [Multi-Stage Continuous Flow — Lag-Aware Quality Forecasting & Drift Monitoring](process-monitoring/multistage-continuous-flow-quality-forecast/) | 1 Hz real production-line telemetry across parallel/serial stages with 15 output measurements per stage | Leakage-safe horizon selection, downstream quality forecasting, conformal uncertainty, and drift diagnostics | Ready |
| 10 | Extrusion Process Monitoring | [Plastic Extrusion — Active-Line Interruption Early Warning](process-monitoring/plastic-extrusion-interruption-early-warning/) | Year-long real extrusion telemetry with 470 variables covering extruders, haul-off, winding, thickness and output | Leakage-safe 20-minute interruption-proxy ranking, frozen-threshold evaluation and drift diagnostics | Ready |


## Advanced learning extensions

Two existing projects now include label- and shift-efficient learning benchmarks without creating synthetic standalone demos:

- **Bosch Production Line:** random, uncertainty, diversity, and hybrid active-learning acquisition curves on the chronological quality-monitoring problem, plus a controlled k-per-class few-label adaptation benchmark. The untouched future test block is never queried.
- **Rotating-Shaft Unbalance:** source-only transfer, unsupervised CORAL, target-only few-shot adaptation, and CORAL + few-shot adaptation. The target session is split chronologically into an adaptation prefix and a later evaluation suffix for this extension.

These extensions are evaluation layers on the existing industrial datasets. They do not change the original verified benchmark claims or reinterpret cross-session transfer as cross-machine generalization.

## Inclusion criteria

Projects are included only when they have a clear industrial data or cyber-physical context. Preferred datasets contain one or more of the following:

- time-series measurements from industrial sensors;
- vibration, temperature, pressure, current, voltage, acoustic, force, torque, flow, or similar process signals;
- PLC, SCADA, historian, MES, CNC-controller, or machine-event data;
- IIoT telemetry and equipment-state information;
- condition-monitoring or predictive-maintenance signals;
- production-process variables suitable for monitoring, diagnosis, control, or optimization;
- automated test/inspection measurements tied to real production lots, timestamps, assets, or traceability;
- multimodal industrial data where vision is tied to real production-system telemetry or traceability.

Synthetic or purely illustrative computer-vision exercises without a meaningful cyber-physical or industrial telemetry layer are excluded.

## Repository principles

- Preserve the raw industrial data structure when redistribution is practical and permitted.
- Keep train/test boundaries aligned with physical experiments, machines, batches, lots, sessions, or time periods to avoid leakage.
- Treat data quality, operating regimes, and process context as first-class parts of the model.
- Report operational metrics at the physical asset/run/lot/session level, not only at the individual-row level.
- Include reproducible training, evaluation, inference/monitoring, and tests for every project.
