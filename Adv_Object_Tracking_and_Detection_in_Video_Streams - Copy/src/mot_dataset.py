"""MOTChallenge readers and a Faster R-CNN training dataset.

MOT17 ground truth uses: frame, identity, left, top, width, height, mark,
class, visibility.  We train a single foreground class (pedestrian, class 1).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _mot_rows(path: Path) -> list[list[float]]:
    """Read a comma-separated MOT text file while ignoring blank lines."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([float(value) for value in row])
    return rows


def read_ground_truth(
    gt_path: Path, min_visibility: float = 0.0
) -> dict[int, list[dict[str, float]]]:
    """Return valid pedestrian annotations grouped by video frame number."""
    by_frame: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in _mot_rows(gt_path):
        if len(row) < 9:
            continue
        frame, identity, left, top, width, height, mark, class_id, visibility = row[:9]
        if int(mark) != 1 or int(class_id) != 1 or visibility < min_visibility:
            continue
        if width <= 1 or height <= 1:
            continue
        by_frame[int(frame)].append(
            {
                "id": int(identity),
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "visibility": visibility,
            }
        )
    return dict(by_frame)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [left, top, width, height] boxes to [x1, y1, x2, y2]."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    result = boxes.clone()
    result[:, 2] = boxes[:, 0] + boxes[:, 2]
    result[:, 3] = boxes[:, 1] + boxes[:, 3]
    return result


class MOTFrameDataset(Dataset):
    """A frame-level pedestrian dataset for Torchvision Faster R-CNN.

    Faster R-CNN performs ImageNet normalization and min/max resizing internally.
    This dataset applies the requested random horizontal flip and colour jitter only
    while training, and keeps bounding boxes synchronized with flips.
    """

    def __init__(
        self,
        dataset_root: Path,
        sequences: list[str],
        frame_stride: int = 1,
        min_visibility: float = 0.0,
        training: bool = False,
    ) -> None:
        self.samples: list[tuple[Path, list[dict[str, float]], int]] = []
        self.training = training
        for sequence in sequences:
            sequence_dir = dataset_root / "train" / sequence
            gt_file = sequence_dir / "gt" / "gt.txt"
            image_dir = sequence_dir / "img1"
            if not gt_file.exists() or not image_dir.exists():
                raise FileNotFoundError(
                    f"Missing {sequence}. Expected {gt_file} and {image_dir}. "
                    "Check paths.dataset_root in config.yaml."
                )
            annotations = read_ground_truth(gt_file, min_visibility)
            for frame in sorted(annotations):
                if (frame - 1) % frame_stride == 0:
                    image_path = image_dir / f"{frame:06d}.jpg"
                    if image_path.exists():
                        self.samples.append((image_path, annotations[frame], frame))
        if not self.samples:
            raise RuntimeError("No labelled frames found. Check sequence names and frame stride.")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        # A gentle random crop simulates partial views.  Boxes are translated,
        # clipped to the crop, and removed only when no visible area remains.
        if torch.rand(()) < 0.4:
            original_image, original_boxes = image, boxes
            width, height = image.size
            crop_width = int(width * float(torch.empty(1).uniform_(0.85, 1.0)))
            crop_height = int(height * float(torch.empty(1).uniform_(0.85, 1.0)))
            left = int(torch.randint(0, max(1, width - crop_width + 1), (1,)))
            top = int(torch.randint(0, max(1, height - crop_height + 1), (1,)))
            cropped = boxes.clone()
            cropped[:, [0, 2]] = (cropped[:, [0, 2]] - left).clamp(0, crop_width)
            cropped[:, [1, 3]] = (cropped[:, [1, 3]] - top).clamp(0, crop_height)
            keep = (cropped[:, 2] - cropped[:, 0] > 1) & (cropped[:, 3] - cropped[:, 1] > 1)
            if keep.any():
                image = TF.crop(image, top, left, crop_height, crop_width)
                boxes = cropped[keep]
            else:
                image, boxes = original_image, original_boxes
        if torch.rand(()) < 0.5:
            width, _ = image.size
            image = TF.hflip(image)
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            boxes = flipped
        # Mild image-only colour jitter; boxes remain geometrically valid.
        image = ImageEnhance.Brightness(image).enhance(float(torch.empty(1).uniform_(0.85, 1.15)))
        image = ImageEnhance.Contrast(image).enhance(float(torch.empty(1).uniform_(0.85, 1.15)))
        image = ImageEnhance.Color(image).enhance(float(torch.empty(1).uniform_(0.85, 1.15)))
        return image, boxes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path, annotations, frame = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        boxes = torch.tensor(
            [[item["left"], item["top"], item["width"], item["height"]] for item in annotations],
            dtype=torch.float32,
        )
        boxes = xywh_to_xyxy(boxes)
        if self.training:
            image, boxes = self._augment(image, boxes)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "frame": torch.tensor([frame]),
        }
        return TF.to_tensor(image), target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Keep variable-sized images and annotation counts in lists for detection models."""
    return tuple(zip(*batch))
