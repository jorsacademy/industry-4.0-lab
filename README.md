# Industry 4.0 Lab

A compact portfolio of applied Industry 4.0 projects focused on smart manufacturing, industrial AI, operations research, simulation, quality engineering, and decision support.

## Project map

| # | Area | Project | Problem | Status |
|---|---|---|---|---|
| 01 | Computer Vision | [Visual Anomaly Detection](computer-vision/visual-anomaly-detection/) | Detect previously unseen manufacturing defects using normal-only training | Ready |

## Design principles

- Each project is self-contained and reproducible.
- Industrial context comes before model complexity.
- Validation includes operational metrics, not only model loss.
- Raw datasets, model weights, and generated artifacts stay out of Git.
- Projects are structured so stronger methods can replace the baseline without changing the surrounding workflow.

## Repository layout

```text
industry-4.0-lab/
├── computer-vision/
│   └── visual-anomaly-detection/
├── optimization/
├── simulation/
├── predictive-maintenance/
├── industrial-iot/
└── docs/
```

New projects will be added incrementally under the relevant Industry 4.0 domain.
