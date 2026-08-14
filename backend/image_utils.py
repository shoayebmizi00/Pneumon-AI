from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from training.dataset import build_transform


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class InvalidImage(ValueError):
    pass


@dataclass(slots=True)
class PreparedImage:
    original: Image.Image
    tensor: torch.Tensor
    domain_warning: str | None


def decode_and_prepare(data: bytes, content_type: str | None, image_size: int) -> PreparedImage:
    if content_type not in ALLOWED_MIME_TYPES:
        raise InvalidImage("Unsupported file type. Upload a JPG, JPEG, or PNG image.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise InvalidImage("The file content is not a supported JPG or PNG image.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as source:
            original = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, InvalidImage):
            raise
        raise InvalidImage("The uploaded image is corrupted or unreadable.") from exc
    if original.width < 128 or original.height < 128:
        raise InvalidImage("Image resolution is too small. Use an X-ray at least 128 × 128 pixels.")
    pixels = np.asarray(original, dtype=np.float32) / 255.0
    channel_difference = float(np.mean(np.max(pixels, axis=2) - np.min(pixels, axis=2)))
    brightness = float(pixels.mean())
    warning = None
    if channel_difference > 0.10 or not 0.08 <= brightness <= 0.92:
        warning = "This image may not resemble the grayscale chest X-rays used for model development. Treat the result as out-of-domain and do not rely on it."
    tensor = build_transform(image_size, training=False)(original).unsqueeze(0)
    return PreparedImage(original, tensor, warning)

