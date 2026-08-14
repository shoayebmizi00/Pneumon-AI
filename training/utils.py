from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path, default: dict | None = None) -> dict:
    target = Path(path)
    if not target.exists():
        return {} if default is None else default
    return json.loads(target.read_text(encoding="utf-8"))


class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best = loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

