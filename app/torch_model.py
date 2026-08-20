"""PyTorch model ta'riflari — faqat BACKEND=torch bo'lganda kerak.

Bu fayl 1-loyihadagi `src/models.py` ning inference uchun zarur qismi.
Nusxalanganining sababi: xizmat o'qitish repozitoriysiga bog'liq
bo'lmasligi kerak — u faqat checkpoint va labels.json bilan ishlaydi.
"""
from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.35) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "scratch":
        return SmallCNN(num_classes)
    if arch in ("resnet18", "resnet34"):
        from torchvision import models

        # weights=None: og'irliklar checkpoint'dan yuklanadi, ImageNet'ni
        # yuklab o'tirish shart emas (offline konteynerda bu muhim).
        model = getattr(models, arch)(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"noma'lum arxitektura: {arch}")
