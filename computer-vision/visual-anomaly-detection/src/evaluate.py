from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .data import ImageDataset, discover_images
from .model import ConvAutoencoder, reconstruction_scores


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(config_path: str | Path, value: str) -> Path:
    return Path(config_path).resolve().parent / value


@torch.inference_mode()
def score_dataset(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    scores: list[np.ndarray] = []
    paths: list[str] = []

    for images, batch_paths in loader:
        images = images.to(device)
        reconstructions = model(images)
        batch_scores = reconstruction_scores(images, reconstructions)
        scores.append(batch_scores.cpu().numpy())
        paths.extend(batch_paths)

    return np.concatenate(scores), paths


def main(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = resolve_project_path(config_path, config["output"]["directory"])
    checkpoint_path = output_dir / config["output"]["checkpoint"]

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ConvAutoencoder(
        latent_channels=int(checkpoint["latent_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(checkpoint["image_size"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])

    normal_paths = discover_images(
        resolve_project_path(config_path, config["data"]["test_normal"])
    )
    anomaly_paths = discover_images(
        resolve_project_path(config_path, config["data"]["test_anomaly"])
    )

    normal_loader = DataLoader(
        ImageDataset(normal_paths, image_size=image_size, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    anomaly_loader = DataLoader(
        ImageDataset(anomaly_paths, image_size=image_size, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    normal_scores, normal_names = score_dataset(model, normal_loader, device)
    anomaly_scores, anomaly_names = score_dataset(model, anomaly_loader, device)

    y_true = np.concatenate(
        [
            np.zeros(len(normal_scores), dtype=int),
            np.ones(len(anomaly_scores), dtype=int),
        ]
    )
    y_score = np.concatenate([normal_scores, anomaly_scores])
    threshold = float(checkpoint["threshold"])
    y_pred = (y_score >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "normal_test_images": len(normal_scores),
        "anomaly_test_images": len(anomaly_scores),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    rows = []
    for path, score in zip(normal_names, normal_scores, strict=True):
        rows.append((path, 0, float(score), int(score >= threshold)))
    for path, score in zip(anomaly_names, anomaly_scores, strict=True):
        rows.append((path, 1, float(score), int(score >= threshold)))

    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "score", "prediction"])
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
