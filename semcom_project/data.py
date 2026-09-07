"""Kodak datasets with an image-level train/validation/test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_split(split_file: str | Path) -> dict:
    with Path(split_file).open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    subsets = [set(split[name]) for name in ("train", "val", "test")]
    if subsets[0] & subsets[1] or subsets[0] & subsets[2] or subsets[1] & subsets[2]:
        raise ValueError("train/val/test split contains overlapping image names")
    return split


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class KodakPatchDataset(Dataset):
    """Generate deterministic random training patches from training images.

    Each epoch changes the crops, but ``seed``, ``epoch`` and ``idx`` uniquely
    determine the selected image and crop. This gives augmentation without
    allowing validation or test pixels into the training set.
    """

    def __init__(
        self,
        image_dir: str | Path,
        image_names: list[str],
        patch_size: int = 64,
        samples_per_epoch: int = 256,
        seed: int = 42,
        horizontal_flip: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_names = list(image_names)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.horizontal_flip = horizontal_flip
        self.epoch = 0
        self.images = []
        for name in self.image_names:
            path = self.image_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
            self.images.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx)
        image = self.images[int(rng.integers(0, len(self.images)))]
        height, width, _ = image.shape
        if min(height, width) < self.patch_size:
            raise ValueError(f"patch_size={self.patch_size} exceeds image shape {image.shape}")
        top = int(rng.integers(0, height - self.patch_size + 1))
        left = int(rng.integers(0, width - self.patch_size + 1))
        patch = image[top : top + self.patch_size, left : left + self.patch_size]
        if self.horizontal_flip and rng.random() < 0.5:
            patch = patch[:, ::-1]
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(patch)


class KodakImageDataset(Dataset):
    """Return full-resolution Kodak images one at a time."""

    def __init__(self, image_dir: str | Path, image_names: list[str]) -> None:
        self.image_dir = Path(image_dir)
        self.image_names = list(image_names)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        name = self.image_names[idx]
        image = Image.open(self.image_dir / name).convert("RGB")
        return image_to_tensor(image), name
