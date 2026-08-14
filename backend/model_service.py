from __future__ import annotations

import threading
from pathlib import Path

import torch

from backend.image_utils import decode_and_prepare
from training.models import create_model, gradcam_target_layer
from training.utils import read_json


class ModelUnavailable(RuntimeError):
    pass


class PneumoniaModelService:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path, self.metadata_path = model_path, metadata_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.metadata: dict = {}
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        try:
            self.metadata = read_json(self.metadata_path)
            if not self.model_path.exists() or not self.metadata:
                raise FileNotFoundError("Trained model artifacts are not present. Run training and evaluation first.")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model = create_model(self.metadata["model"], pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device).eval()
            self.model, self.load_error = model, None
        except Exception as exc:
            self.model, self.load_error = None, str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, data: bytes, content_type: str | None, with_gradcam: bool = False) -> dict:
        size = int(self.metadata.get("image_size", [224, 224])[0])
        prepared = decode_and_prepare(data, content_type, size)
        if not self.ready:
            raise ModelUnavailable(self.load_error or "Model is unavailable.")
        tensor = prepared.tensor.to(self.device)
        with self._lock:
            with torch.inference_mode():
                logit = float(self.model(tensor).item())
            temperature = float(self.metadata.get("temperature", 1.0))
            pneumonia_score = float(torch.sigmoid(torch.tensor(logit / temperature)).item())
            threshold = float(self.metadata["threshold"])
            margin = float(self.metadata.get("uncertainty_margin", 0.08))
            prediction = "PNEUMONIA" if pneumonia_score >= threshold else "NORMAL"
            uncertain = abs(pneumonia_score - threshold) <= margin or prepared.domain_warning is not None
            gradcam_uri = None
            if with_gradcam:
                # OpenCV and Grad-CAM add substantial memory overhead. Import them
                # only for deployments that explicitly request an attention map.
                import base64

                from training.gradcam import GradCAM, overlay_heatmap

                cam = GradCAM(self.model, gradcam_target_layer(self.model, self.metadata["model"]))
                try:
                    heatmap = cam.generate(tensor)
                    gradcam_uri = "data:image/png;base64," + base64.b64encode(overlay_heatmap(prepared.original, heatmap)).decode("ascii")
                finally:
                    cam.close()
        confidence = pneumonia_score if prediction == "PNEUMONIA" else 1 - pneumonia_score
        return {
            "success": True, "prediction": prediction, "prediction_score": round(confidence, 6),
            "normal_score": round(1 - pneumonia_score, 6), "pneumonia_score": round(pneumonia_score, 6),
            "decision_threshold": threshold, "uncertain": uncertain, "domain_warning": prepared.domain_warning,
            "gradcam_image": gradcam_uri,
            "disclaimer": "Educational/research prototype only. This is not a medical diagnosis.",
        }
