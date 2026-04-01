"""Dataset classes for the SupCon pipeline.

RaptorDataset  — 20 000 pre-cropped IR module images (12 classes)
UAVPanelDataset — panel patches extracted from Zenodo UAV thermal frames

Following Bommes et al. (arXiv:2112.02922):
  - Images loaded as grayscale, resized to 64×64
  - Optionally standardised per-dataset (subtract mean, divide by std)
  - Stacked to 3-channel inside the encoder (x.expand(-1, 3, -1, -1))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lib.utils import CLASS_NAMES_12


class RaptorDataset(Dataset):
    """Raptor Maps InfraredSolarModules (grayscale, 24×40 px originals)."""

    def __init__(
        self,
        images_dir: Path,
        metadata_path: Path,
        transform: Callable | None = None,
        indices: list[int] | None = None,
        standardize: bool = False,
        mean: float | None = None,
        std: float | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.standardize = standardize
        self._mean = mean
        self._std = std

        with open(metadata_path) as f:
            meta = json.load(f)

        self._name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES_12)}

        self.samples: list[tuple[Path, int]] = []
        for entry in meta.values():
            img_path = self.images_dir / Path(entry["image_filepath"]).name
            label = self._name_to_idx[entry["anomaly_class"]]
            self.samples.append((img_path, label))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def compute_stats(self) -> tuple[float, float]:
        """Compute dataset mean and std for standardisation (Bommes §IV-C.2)."""
        if self._mean is not None and self._std is not None:
            return self._mean, self._std
        pixel_sum, pixel_sq_sum, n = 0.0, 0.0, 0
        for img_path, _ in self.samples:
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
            pixel_sum += img.sum()
            pixel_sq_sum += (img ** 2).sum()
            n += img.size
        mean = pixel_sum / n
        std = np.sqrt(pixel_sq_sum / n - mean ** 2)
        self._mean = float(mean / 255.0)
        self._std = float(std / 255.0)
        return self._mean, self._std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = img.resize((64, 64), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        if self.standardize and self._mean is not None and self._std is not None:
            img = (img - self._mean) / max(self._std, 1e-6)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> list[int]:
        return [s[1] for s in self.samples]


class UAVPanelDataset(Dataset):
    """Panel crops extracted from Zenodo UAV thermal frames."""

    def __init__(
        self,
        crops_dir: Path,
        transform: Callable | None = None,
        standardize: bool = False,
        mean: float | None = None,
        std: float | None = None,
    ):
        self.crops_dir = Path(crops_dir)
        self.transform = transform
        self.standardize = standardize
        self._mean = mean
        self._std = std
        self.paths = sorted(self.crops_dir.rglob("*.jpg"))
        if not self.paths:
            self.paths = sorted(self.crops_dir.rglob("*.png"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("L")
        img = img.resize((64, 64), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        if self.standardize and self._mean is not None and self._std is not None:
            img = (img - self._mean) / max(self._std, 1e-6)
        if self.transform is not None:
            img = self.transform(img)
        return img, str(self.paths[idx])
