"""Faster R-CNN model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(config: dict, pretrained: bool = True) -> torch.nn.Module:
    """Create Faster R-CNN with a ResNet-50 FPN backbone and two output classes."""
    model_cfg = config["model"]
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=int(model_cfg["min_size"]),
        max_size=int(model_cfg["max_size"]),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model


def load_detector(checkpoint_path: Path, config: dict, device: torch.device) -> torch.nn.Module:
    """Load a trained checkpoint into the exact architecture used for fine-tuning."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run src/train_detector.py first."
        )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_detector(config, pretrained=False)
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval()
