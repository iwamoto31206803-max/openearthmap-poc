"""GSI paddy class-7 positive-only partial supervision; 255 is never negative."""

from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.model import rgb_to_tensor
from src.training.prepare_gsi_labels import IGNORE_INDEX, _png_index, _sha256

TARGET_CLASS = 7
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Sample:
    source_image_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    positive_pixel_count: int
    image_sha256: str
    label_sha256: str


def read_label(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        label = np.array(image)  # Do not convert palette indices or 255 to RGB.
    if label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError(f"Expected a single-channel uint8 label: {path.name}")
    if not np.all((label == TARGET_CLASS) | (label == IGNORE_INDEX)):
        raise ValueError(f"Paddy labels must contain only 7 and 255: {path.name}")
    return label


def scan_dataset(org_dir: Path, labels_dir: Path) -> tuple[list[Sample], list[Sample]]:
    """Audit every pair, retaining paths/counts/hashes only, not image arrays."""
    images = _png_index(Path(org_dir), "org")
    labels = _png_index(Path(labels_dir), "labels")
    if not images or images.keys() != labels.keys():
        raise ValueError("PNG pairing failed: org and labels must have identical relative paths")
    usable, excluded = [], []
    for relative in sorted(images):
        with Image.open(images[relative]) as image:
            width, height = image.size
            image.verify()
        label = read_label(labels[relative])
        if label.shape != (height, width):
            raise ValueError(f"Image size mismatch for {relative}")
        positives = int(np.count_nonzero(label == TARGET_CLASS))
        sample = Sample(
            Path(relative).with_suffix("").as_posix(), images[relative], labels[relative],
            width, height, positives, _sha256(images[relative]), _sha256(labels[relative]),
        )
        (usable if positives else excluded).append(sample)
    return usable, excluded


def split_samples(samples: list[Sample], train_ratio: float = 0.8,
                  seed: int = DEFAULT_SEED) -> tuple[list[Sample], list[Sample]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
    if len(samples) < 2:
        raise ValueError("At least two positive images are needed for train/validation")
    ordered = sorted(samples, key=lambda sample: sample.source_image_id)
    if len({sample.source_image_id for sample in ordered}) != len(ordered):
        raise ValueError("Duplicate source image IDs")
    random.Random(seed).shuffle(ordered)
    count = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio)))
    return ordered[:count], ordered[count:]


class GsiPaddyDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        label = read_label(sample.label_path)
        if rgb.shape[:2] != label.shape:
            raise ValueError(f"Image size mismatch for {sample.source_image_id}")
        if not np.any(label == TARGET_CLASS):
            raise ValueError("All-ignore image reached training; rescan the dataset")
        return rgb_to_tensor(rgb), torch.from_numpy(label.astype(np.int64))


def collate_padded(batch):
    """Pad right/bottom to batch maximum rounded to encoder stride 32.

    No resize/crop: every labeled pixel survives. RGB uses edge replication as
    in tiled inference. Added label pixels are ignored by both loss and metrics.
    """
    height = ((max(rgb.shape[1] for rgb, _ in batch) + 31) // 32) * 32
    width = ((max(rgb.shape[2] for rgb, _ in batch) + 31) // 32) * 32
    images, labels = [], []
    for rgb, label in batch:
        padding = (0, width - rgb.shape[2], 0, height - rgb.shape[1])
        images.append(F.pad(rgb, padding, mode="replicate"))
        labels.append(F.pad(label, padding, value=IGNORE_INDEX))
    return torch.stack(images), torch.stack(labels)
