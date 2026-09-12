from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from .model import ConvAutoencoder, reconstruction_scores, residual_map


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(config_path: str | Path, value: str) -> Path:
    return Path(config_path).resolve().parent / value


def main(image_path: str, config_path: str) -> None:
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

    image_size = int(checkpoint["image_size"])
    model = ConvAutoencoder(
        latent_channels=int(checkpoint["latent_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        reconstruction = model(tensor)
        score = float(reconstruction_scores(tensor, reconstruction).item())
        heatmap = residual_map(tensor, reconstruction).squeeze(0).cpu().numpy()

    threshold = float(checkpoint["threshold"])
    decision = "ANOMALY" if score >= threshold else "NORMAL"

    input_image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstruction_image = (
        reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(input_image, 0.0, 1.0))
    axes[0].set_title("Input")
    axes[1].imshow(np.clip(reconstruction_image, 0.0, 1.0))
    axes[1].set_title("Reconstruction")
    axes[2].imshow(heatmap, cmap="inferno")
    axes[2].set_title("Residual")
    for axis in axes:
        axis.axis("off")

    fig.suptitle(f"{decision} | score={score:.6f} | threshold={threshold:.6f}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_inspection.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"decision={decision}")
    print(f"score={score:.8f}")
    print(f"threshold={threshold:.8f}")
    print(f"inspection={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.image, args.config)
