from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best_pneumonia_model.pth"))
    metadata_path: Path = Path(os.getenv("METADATA_PATH", ROOT / "models" / "model_metadata.json"))
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))
    allowed_origins: tuple[str, ...] = tuple(filter(None, os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")))
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "30"))


settings = Settings()

