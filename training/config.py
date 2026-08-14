from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ExperimentConfig:
    dataset_dir: str = str(ROOT / "pneumonia_dataset" / "chest_xray")
    output_dir: str = str(ROOT / "artifacts")
    image_size: int = 224
    batch_size: int = 24
    seed: int = 42
    workers: int = 0
    models: Sequence[str] = ("baseline", "densenet121", "efficientnet_b0", "resnet50", "mobilenet_v3")
    frozen_epochs: int = 8
    finetune_epochs: int = 8
    frozen_lr: float = 1e-3
    finetune_lr: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 3
    min_sensitivity: float = 0.85
    uncertainty_margin: float = 0.08
    pretrained: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

