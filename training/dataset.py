from __future__ import annotations

import re
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASS_TO_INDEX = {"NORMAL": 0, "PNEUMONIA": 1}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def patient_id_from_filename(path: str | Path) -> str:
    """Extract the best available patient identifier from this dataset's names."""
    stem = Path(path).stem.lower()
    person = re.match(r"(person\d+)", stem)
    if person:
        return person.group(1)
    normal = re.match(r"(im-\d+)", stem)
    if normal:
        return normal.group(1)
    return stem.split("_")[0]


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str
    split: str
    patient_id: str


def discover_records(dataset_dir: str | Path, splits: Iterable[str] = ("train", "val", "test")) -> list[ImageRecord]:
    root = Path(dataset_dir)
    records: list[ImageRecord] = []
    for split in splits:
        for class_name, label in CLASS_TO_INDEX.items():
            folder = root / split / class_name
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                    records.append(ImageRecord(path, label, class_name, split, patient_id_from_filename(path)))
    return records


def load_manifest(path: str | Path) -> list[ImageRecord]:
    manifest = Path(path)
    records: list[ImageRecord] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(ImageRecord(Path(row["path"]), int(row["label"]), row["class_name"], row["split"], row["patient_id"]))
    return records


def build_transform(image_size: int, training: bool) -> transforms.Compose:
    steps: list = [transforms.Grayscale(num_output_channels=3)]
    if training:
        steps.extend([
            transforms.RandomAffine(degrees=5, translate=(0.025, 0.025), scale=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    steps.extend([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(steps)


class ChestXrayDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, torch.tensor(record.label, dtype=torch.float32), str(record.path)


def make_loaders(dataset_dir: str | Path, image_size: int, batch_size: int, workers: int = 0, manifest: str | Path | None = None) -> tuple[dict[str, DataLoader], dict[str, int]]:
    records = load_manifest(manifest) if manifest else discover_records(dataset_dir)
    by_split = {split: [r for r in records if r.split == split] for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(
            ChestXrayDataset(items, build_transform(image_size, training=split == "train")),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, items in by_split.items()
    }
    counts = Counter(r.class_name for r in by_split["train"])
    return loaders, dict(counts)
