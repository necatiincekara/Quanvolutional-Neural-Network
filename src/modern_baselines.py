"""
Modern compact classical baselines for the publication benchmark stack.

These models are intentionally kept separate from thesis-faithful reproductions
and current-local matched-budget ablations. Their role is reviewer-proofing:
they test whether a stronger contemporary classical architecture materially
changes the paper's benchmark story.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import models, transforms


def resnet18_cifar_gray_transform() -> transforms.Compose:
    """
    Conservative augmentation for 32x32 grayscale characters.

    Horizontal flips are intentionally excluded because they would alter the
    semantics of handwritten characters. Affine jitter is kept modest.
    """
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
                shear=8,
                fill=0.0,
            )
        ]
    )


class ResNet18CIFARGray(nn.Module):
    """
    ResNet18 adapted for 32x32 single-channel inputs.

    Changes relative to the torchvision ImageNet stem:
    - conv1: 7x7 stride 2 -> 3x3 stride 1
    - maxpool removed
    - input channels: 3 -> 1
    """

    def __init__(self, num_classes: int = 44):
        super().__init__()
        backbone = models.resnet18(weights=None, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


MODERN_BASELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18_cifar_gray": {
        "builder": ResNet18CIFARGray,
        "family": "modern-classical",
        "source": "modern-baseline",
        "default_epochs": 60,
        "batch_size": 128,
        "train_transform": resnet18_cifar_gray_transform(),
        "notes": (
            "Reviewer-proof stronger classical baseline: torchvision ResNet18 "
            "adapted to 32x32 grayscale inputs with a CIFAR-style stem."
        ),
    }
}


def create_modern_baseline(model_name: str, num_classes: int = 44) -> nn.Module:
    return MODERN_BASELINE_REGISTRY[model_name]["builder"](num_classes=num_classes)
