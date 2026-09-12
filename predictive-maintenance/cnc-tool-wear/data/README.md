# CNC Telemetry Data

The project uses 18 complete milling experiments sampled every 100 ms (10 Hz). Run-level labels and operating settings are stored in `raw/train.csv`; each `raw/experiment_XX.csv` contains one experiment's controller time series.

## Run-level metadata

| Field | Meaning |
|---|---|
| `No` | Experiment identifier (1-18) |
| `material` | Workpiece material |
| `feedrate` | Configured feed rate |
| `clamp_pressure` | Workholding pressure |
| `tool_condition` | `worn` or `unworn` |
| `machining_finalized` | Whether the machining run completed |
| `passed_visual_inspection` | Final inspection result when available |

## Time-series channels

The controller stream contains motion-command, motion-feedback and electrical measurements for the X, Y and Z servo axes and the spindle (`S1`). Depending on the axis, channels include:

- actual and commanded position;
- actual and commanded velocity;
- actual and commanded acceleration;
- current feedback;
- DC-bus voltage;
- output current;
- output voltage;
- output power;
- spindle system inertia.

The controller/program context includes:

- `M1_CURRENT_PROGRAM_NUMBER`;
- `M1_sequence_number`;
- `M1_CURRENT_FEEDRATE`;
- `Machining_Process`.

## Process states

`Machining_Process` identifies the active stage of the CNC cycle, including preparation, repositioning, and the individual cutting layers. It is retained as process context rather than flattened away.

## Data-quality rule

Some controller states are known to be unreliable. The project flags rows where the current feed rate is 50, the X-axis actual position is 198, or the current program number is non-zero. By default these rows are removed before window construction; the behavior is configurable in `config.yaml`.

## Integrity

After the raw files are hydrated, `raw/manifest.json` contains a SHA-256 digest and byte size for every original data file. `scripts/validate_data.py` verifies the 18 experiment files, the run metadata, and the common high-dimensional telemetry schema.
