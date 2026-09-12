# Visual Anomaly Detection for Manufactured Parts

A normal-only visual inspection pipeline for smart manufacturing. The model learns the appearance of conforming parts and treats reconstruction deviation as an anomaly signal, allowing it to flag defect types that were not available during training.

## Industrial objective

In many production lines, good parts are abundant while defect examples are sparse, changing, or unavailable during commissioning. A supervised defect classifier is therefore often the wrong starting point. This project uses one-class learning:

- training uses only conforming images;
- validation uses only conforming images to set the operating threshold;
- defective images are used only for final evaluation;
- inference returns both an anomaly score and a visual residual map.

This mirrors an early-stage automated quality gate where the system must detect unknown defects while tolerating normal variation in position, orientation, and illumination.

## Method

The baseline is a denoising convolutional autoencoder.

1. Normal images are resized and mildly augmented to model expected process variation.
2. Gaussian noise is added during training, while the network reconstructs the clean normal image.
3. Per-image reconstruction error becomes the anomaly score.
4. The decision threshold is estimated exclusively from normal validation scores using a high quantile.
5. Final test performance is reported with ROC-AUC, average precision, precision, recall, F1, and the confusion matrix.

The separation between score generation and threshold calibration makes the project easy to extend later with PatchCore, PaDiM, teacher-student models, or geometric inspection features.

## Project structure

```text
visual-anomaly-detection/
├── config.yaml
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
└── tests/
    └── test_model.py
```

## Data layout

Place images locally using this structure:

```text
data/raw/
├── train/
│   └── normal/
├── test/
│   ├── normal/
│   └── anomaly/
```

The raw image folders are intentionally ignored by Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config config.yaml
```

Training writes the model checkpoint and normal-validation threshold to `artifacts/`.

## Evaluate

```bash
python -m src.evaluate --config config.yaml
```

The evaluation command writes:

- `artifacts/metrics.json`
- `artifacts/scores.csv`

The decision threshold is never optimized on the anomalous test set.

## Inspect one image

```bash
python -m src.infer path/to/image.png --config config.yaml
```

The command prints the anomaly score and decision, then saves a three-panel inspection image containing the input, reconstruction, and residual heatmap.

## Operational interpretation

A production deployment should treat the threshold as an operating point rather than a universal constant. The preferred threshold depends on the cost of a false reject versus the cost of allowing a defective part downstream. Drift monitoring should also be added when camera, lighting, material, tooling, or process settings change.

## Next improvements

- patch-level feature memory for local defect sensitivity;
- geometric alignment before scoring;
- threshold calibration by target false-reject rate;
- ONNX export for edge inference;
- production drift monitoring;
- latency and throughput benchmarking.
