# Sources and provenance

## Primary dataset

Kaggle dataset: **CNC turning: roughness, forces and tool wear**

- Dataset page: https://www.kaggle.com/datasets/adorigueto/cnc-turning-roughness-forces-and-tool-wear
- Kaggle dataset DOI: `10.34740/kaggle/ds/2205074`
- License: **CC BY-NC-SA 4.0**

The dataset was produced in the Competence Center in Manufacturing at the Aeronautics Institute of Technology in Brazil as part of a master's research project.

Public documentation reports two planned experiments on AISI H13 steel with cutting fluid:

- `Exp1.csv`: 324 rows = 54 machining runs × 6 roughness positions;
- `Exp2.csv`: 288 rows = 48 machining runs × 6 roughness positions.

Measured variables include machining setpoints, surface-roughness metrics, three force components/resultant force, and flank-wear width (`TCond`). Forces were acquired with a Kistler dynamometer system; surface roughness with a Mitutoyo Surftest SJ-210; tool wear with a digital microscope.

## Implementation policy

This repository contains an independent implementation. It does not copy Kaggle notebooks or the source author's ANN code. The raw dataset is downloaded at use time and not redistributed here. Derived benchmark reports contain model outputs and aggregate summaries rather than a substitute copy of the source tables.
