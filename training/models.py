from __future__ import annotations

import torch
from torch import nn
from torchvision import models


MODEL_NAMES = ("baseline", "densenet121", "efficientnet_b0", "resnet50", "mobilenet_v3")


class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (3, 32, 64, 128, 256)
        blocks: list[nn.Module] = []
        for incoming, outgoing in zip(channels, channels[1:]):
            blocks.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x))).squeeze(1)


def _weights(enabled: bool, enum):
    return enum.DEFAULT if enabled else None


def create_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "baseline":
        return BaselineCNN()
    if name == "densenet121":
        model = models.densenet121(weights=_weights(pretrained, models.DenseNet121_Weights))
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier.in_features, 1))
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=_weights(pretrained, models.EfficientNet_B0_Weights))
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, 1))
    elif name == "resnet50":
        model = models.resnet50(weights=_weights(pretrained, models.ResNet50_Weights))
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    elif name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=_weights(pretrained, models.MobileNet_V3_Small_Weights))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
    return model


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool, unfreeze_fraction: float = 0.25) -> None:
    if name == "baseline":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.classifier if hasattr(model, "classifier") else model.fc
    for parameter in head.parameters():
        parameter.requires_grad = True
    if trainable:
        feature_module = model.features if hasattr(model, "features") else nn.Sequential(*list(model.children())[:-1])
        children = list(feature_module.children())
        start = max(0, int(len(children) * (1 - unfreeze_fraction)))
        for child in children[start:]:
            for parameter in child.parameters():
                parameter.requires_grad = True


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    if name == "densenet121":
        return model.features.denseblock4
    if name == "efficientnet_b0":
        return model.features[-1]
    if name == "resnet50":
        return model.layer4[-1]
    if name == "mobilenet_v3":
        return model.features[-1]
    return model.features[-4]
