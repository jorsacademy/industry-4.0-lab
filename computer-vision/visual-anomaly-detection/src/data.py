from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in: {root}")
    return paths


def split_paths(
    paths: Sequence[Path],
    validation_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(paths) < 2:
        raise ValueError("At least two images are required for a train/validation split")

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)

    n_validation = max(1, round(len(shuffled) * validation_ratio))
    n_validation = min(n_validation, len(shuffled) - 1)
    return shuffled[n_validation:], shuffled[:n_validation]


def build_transform(image_size: int, augment: bool) -> transforms.Compose:
    steps: list[object] = [transforms.Resize((image_size, image_size))]

    if augment:
        steps.extend(
            [
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


class ImageDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[Path],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("ImageDataset received no image paths")
        self.transform = build_transform(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, str(path)
