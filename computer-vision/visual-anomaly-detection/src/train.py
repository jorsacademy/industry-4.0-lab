from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import ImageDataset, discover_images, split_paths
from .model import ConvAutoencoder, reconstruction_scores


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(config_path: str | Path, value: str) -> Path:
    return Path(config_path).resolve().parent / value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []

    for images, _ in loader:
        images = images.to(device)
        reconstructions = model(images)
        scores = reconstruction_scores(images, reconstructions)
        values.append(scores.cpu().numpy())

    return np.concatenate(values)


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = resolve_project_path(config_path, config["data"]["train_normal"])
    all_paths = discover_images(train_root)
    train_paths, validation_paths = split_paths(
        all_paths,
        validation_ratio=float(config["training"]["validation_ratio"]),
        seed=seed,
    )

    image_size = int(config["model"]["image_size"])
    train_dataset = ImageDataset(train_paths, image_size=image_size, augment=True)
    validation_dataset = ImageDataset(
        validation_paths, image_size=image_size, augment=False
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ConvAutoencoder(
        latent_channels=int(config["model"]["latent_channels"])
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.MSELoss()
    noise_std = float(config["training"]["noise_std"])
    epochs = int(config["training"]["epochs"])

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for clean_images, _ in train_loader:
            clean_images = clean_images.to(device)
            noisy_images = (
                clean_images + noise_std * torch.randn_like(clean_images)
            ).clamp(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)
            reconstructions = model(noisy_images)
            loss = criterion(reconstructions, clean_images)
            loss.backward()
            optimizer.step()

            batch_n = clean_images.size(0)
            running_loss += loss.item() * batch_n
            sample_count += batch_n

        train_loss = running_loss / sample_count

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.inference_mode():
            for images, _ in validation_loader:
                images = images.to(device)
                reconstructions = model(images)
                loss = criterion(reconstructions, images)
                batch_n = images.size(0)
                validation_loss += loss.item() * batch_n
                validation_count += batch_n

        validation_loss /= validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

    normal_validation_scores = collect_scores(model, validation_loader, device)
    quantile = float(config["training"]["threshold_quantile"])
    threshold = float(np.quantile(normal_validation_scores, quantile))

    output_dir = resolve_project_path(config_path, config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config["output"]["checkpoint"]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "threshold_quantile": quantile,
            "image_size": image_size,
            "latent_channels": int(config["model"]["latent_channels"]),
        },
        checkpoint_path,
    )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"device={device}")
    print(f"train_images={len(train_paths)} validation_images={len(validation_paths)}")
    print(f"threshold={threshold:.8f}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
