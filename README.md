# Industry 4.0 Lab

A curated portfolio of applied Industry 4.0 projects built around cyber-physical production systems, industrial sensing, IIoT, PLC/SCADA data, condition monitoring, predictive maintenance, process analytics, digital twins, and optimization.

## Project map

| # | Domain | Project | Industrial data | Main objective | Status |
|---|---|---|---|---|---|
| 01 | Predictive Maintenance | [CNC Tool Wear & Process Health Monitoring](predictive-maintenance/cnc-tool-wear/) | 10 Hz multi-axis CNC servo and spindle telemetry | Detect tool wear and support online condition monitoring | Ready |

## Inclusion criteria

Projects are included only when they have a clear industrial data or cyber-physical context. Preferred datasets contain one or more of the following:

- time-series measurements from industrial sensors;
- vibration, temperature, pressure, current, voltage, acoustic, force, torque, flow, or similar process signals;
- PLC, SCADA, historian, MES, CNC-controller, or machine-event data;
- IIoT telemetry and equipment-state information;
- condition-monitoring or predictive-maintenance signals;
- production-process variables suitable for monitoring, diagnosis, control, or optimization;
- multimodal industrial data where vision is tied to real production-system telemetry or traceability.

Synthetic or purely illustrative computer-vision exercises without a meaningful cyber-physical or industrial telemetry layer are excluded.

## Repository principles

- Preserve the raw industrial data structure when redistribution is practical and permitted.
- Keep train/test boundaries aligned with physical experiments, machines, batches, or time periods to avoid leakage.
- Treat data quality, operating regimes, and process context as first-class parts of the model.
- Report operational metrics at the physical asset/run level, not only at the individual-row level.
- Include reproducible training, evaluation, inference, and tests for every project.
