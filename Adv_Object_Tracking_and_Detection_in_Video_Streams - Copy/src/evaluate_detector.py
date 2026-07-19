"""Recompute validation detection mAP from a saved Faster R-CNN checkpoint."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader

from src.common import get_device, load_config, resolve_path, write_json
from src.detector import load_detector
from src.mot_dataset import MOTFrameDataset, collate_fn
from src.train_detector import evaluate_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config, config_folder = load_config(args.config)
    device = get_device(config["runtime"]["device"])
    root = resolve_path(config["paths"]["dataset_root"], config_folder)
    output_root = resolve_path(config["paths"]["output_root"], config_folder)
    checkpoint = resolve_path(config["paths"]["checkpoint_path"], config_folder)
    data_cfg = config["data"]
    validation = MOTFrameDataset(root, [data_cfg["validation_sequence"]], int(data_cfg["validation_frame_stride"]), float(data_cfg["min_visibility"]), training=False)
    loader = DataLoader(validation, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]), collate_fn=collate_fn)
    metrics = evaluate_map(load_detector(checkpoint, config, device), loader, device)
    metrics.update({"sequence": data_cfg["validation_sequence"], "checkpoint": str(checkpoint), "frames": len(validation)})
    path = output_root / "detection_metrics.json"
    write_json(path, metrics)
    print(metrics)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
