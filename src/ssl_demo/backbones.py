"""Backbone registry for the minimum demo.

``get_backbone(name)`` returns ``(model.eval(), preprocess, embed_dim)``.
Each preprocess uses that model's own normalisation and input size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

# ImageNet-1K stats used by torchvision ResNet-50 IMAGENET1K_V1 and by DINOv2.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PHASE1_BACKBONES = ("random_resnet50", "sup_resnet50_in1k", "dinov2_vits14")


@dataclass(frozen=True)
class BackboneInfo:
    name: str
    embed_dim: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    kind: str
    licence: str
    source: str


BACKBONES: dict[str, BackboneInfo] = {
    "random_resnet50": BackboneInfo(
        name="random_resnet50",
        embed_dim=2048,
        input_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        kind="pooled",
        licence="Randomly initialised torchvision ResNet-50. No pretrained weights.",
        source="torchvision.models.resnet50(weights=None)",
    ),
    "sup_resnet50_in1k": BackboneInfo(
        name="sup_resnet50_in1k",
        embed_dim=2048,
        input_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        kind="pooled",
        licence=(
            "torchvision ResNet-50 IMAGENET1K_V1. The checkpoint is downloaded, "
            "not stored in this repo. ImageNet content has its own terms."
        ),
        source="torchvision ResNet50_Weights.IMAGENET1K_V1",
    ),
    "dinov2_vits14": BackboneInfo(
        name="dinov2_vits14",
        embed_dim=384,
        input_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        kind="cls+patchmean",
        licence="DINOv2 (facebookresearch/dinov2), Apache-2.0. Loaded with torch.hub.",
        source="torch.hub facebookresearch/dinov2 dinov2_vits14",
    ),
}


def list_backbones() -> tuple[str, ...]:
    return tuple(BACKBONES)


def backbone_info(name: str) -> BackboneInfo:
    try:
        return BACKBONES[name]
    except KeyError as exc:
        known = ", ".join(BACKBONES)
        raise KeyError(f"Unknown backbone {name!r}. Known: {known}") from exc


def make_preprocess(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    input_size: int,
) -> transforms.Compose:
    """Resize, center-crop, and normalise with this model's own stats."""
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _imagenet_stats_from_weights(weights: ResNet50_Weights) -> tuple[tuple[float, ...], tuple[float, ...]]:
    preset = weights.transforms()
    return tuple(preset.mean), tuple(preset.std)


def _resnet50(pretrained: bool) -> tuple[nn.Module, tuple[float, ...], tuple[float, ...]]:
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        mean, std = _imagenet_stats_from_weights(weights)
    else:
        model = resnet50(weights=None)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    model.fc = nn.Identity()
    return model, mean, std


def _dinov2() -> nn.Module:
    # trust_repo skips the interactive prompt so a Colab cell does not hang.
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)


def get_backbone(name: str) -> tuple[nn.Module, transforms.Compose, int]:
    """Return an eval-mode model, its preprocess, and the embedding size.

    Supervised ResNet-50 and DINOv2 download weights on first use.
    ``random_resnet50`` does not.
    """
    info = backbone_info(name)
    if name == "random_resnet50":
        model, mean, std = _resnet50(pretrained=False)
    elif name == "sup_resnet50_in1k":
        model, mean, std = _resnet50(pretrained=True)
    elif name == "dinov2_vits14":
        model = _dinov2()
        mean, std = info.mean, info.std
    else:
        raise KeyError(name)
    model.eval()
    preprocess = make_preprocess(mean, std, info.input_size)
    return model, preprocess, info.embed_dim


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
