"""Quick read-only preflight check for paths, required imports, and MOT17 layout."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from src.common import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config, base = load_config(args.config)
    for package in ("torch", "torchvision", "cv2", "motmetrics", "torchmetrics", "scipy", "yaml"):
        importlib.import_module(package)
    root = resolve_path(config["paths"]["dataset_root"], base)
    if not (root / "train").is_dir() or not (root / "test").is_dir():
        raise FileNotFoundError(f"Expected train/ and test/ below {root}")
    sequence = config["data"]["validation_sequence"]
    frame_dir = root / "train" / sequence / "img1"
    gt_file = root / "train" / sequence / "gt" / "gt.txt"
    if not frame_dir.is_dir() or not gt_file.is_file():
        raise FileNotFoundError(f"Missing expected validation files under {root / 'train' / sequence}")
    print("Setup check passed")
    print("Dataset root:", root)
    print("Validation sequence:", sequence)
    print("Frames:", len(list(frame_dir.glob("*.jpg"))))
    print("Output root:", resolve_path(config["paths"]["output_root"], base))


if __name__ == "__main__":
    main()
