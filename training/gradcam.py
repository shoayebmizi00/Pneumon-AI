from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model, self.activations, self.gradients = model, None, None
        self.handles = [target_layer.register_forward_hook(self._forward), target_layer.register_full_backward_hook(self._backward)]

    def _forward(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(tensor)
        logit.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1)).squeeze().cpu().numpy()
        heatmap -= heatmap.min()
        return heatmap / (heatmap.max() + 1e-8)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.38) -> bytes:
    rgb = np.asarray(original.convert("RGB"))
    resized = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    blended = np.clip(rgb * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
