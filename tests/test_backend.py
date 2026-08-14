from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app, service
from backend.config import settings


client = TestClient(app)


def image_bytes(size=(256, 256)) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", size, color=110).save(buffer, "PNG")
    return buffer.getvalue()


def test_health_reports_model_state() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is service.ready


def test_missing_upload_is_rejected() -> None:
    assert client.post("/api/predict").status_code == 422


def test_unsupported_file_is_rejected() -> None:
    response = client.post("/api/predict", files={"file": ("report.pdf", b"%PDF", "application/pdf")})
    assert response.status_code == 415


def test_oversized_upload_is_rejected() -> None:
    payload = b"x" * (settings.max_upload_size + 1)
    response = client.post("/api/predict", files={"file": ("large.png", payload, "image/png")})
    assert response.status_code == 413


def test_corrupt_image_is_rejected_when_model_is_ready(monkeypatch) -> None:
    monkeypatch.setattr(service, "model", object())
    monkeypatch.setattr(service, "metadata", {"image_size": [224, 224]})
    response = client.post("/api/predict", files={"file": ("xray.png", b"not-png", "image/png")})
    assert response.status_code == 415


def test_successful_prediction_contract(monkeypatch) -> None:
    expected = {"success": True, "prediction": "NORMAL", "prediction_score": .8, "normal_score": .8,
                "pneumonia_score": .2, "decision_threshold": .5, "uncertain": False,
                "domain_warning": None, "gradcam_image": None, "disclaimer": "research only"}
    monkeypatch.setattr(service, "predict", lambda *_args, **_kwargs: expected)
    response = client.post("/api/predict", files={"file": ("xray.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == expected
