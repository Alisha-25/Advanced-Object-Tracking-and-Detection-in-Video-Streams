"""Fine-tune Faster R-CNN on MOT17 pedestrian boxes and report validation mAP."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.common import get_device, load_config, resolve_path, seed_everything, write_json
from src.detector import build_detector
from src.mot_dataset import MOTFrameDataset, collate_fn


@torch.inference_mode()
def evaluate_map(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute COCO-style detection mAP on held-out frames."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        metric.update(
            [{key: value.detach().cpu() for key, value in output.items() if key in {"boxes", "scores", "labels"}} for output in outputs],
            [{key: value.detach().cpu() for key, value in target.items() if key in {"boxes", "labels", "area", "iscrowd"}} for target in targets],
        )
    result = metric.compute()
    return {key: float(value) for key, value in result.items() if value.numel() == 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs for a quick smoke test.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use COCO pretrained weights.")
    args = parser.parse_args()

    config, config_folder = load_config(args.config)
    seed_everything(int(config["training"]["seed"]))
    device = get_device(config["runtime"]["device"])
    root = resolve_path(config["paths"]["dataset_root"], config_folder)
    output_root = resolve_path(config["paths"]["output_root"], config_folder)
    checkpoint_path = resolve_path(config["paths"]["checkpoint_path"], config_folder)

    data_cfg = config["data"]
    train_set = MOTFrameDataset(
        root, data_cfg["train_sequences"], int(data_cfg["train_frame_stride"]),
        float(data_cfg["min_visibility"]), training=True,
    )
    validation_set = MOTFrameDataset(
        root, [data_cfg["validation_sequence"]], int(data_cfg["validation_frame_stride"]),
        float(data_cfg["min_visibility"]), training=False,
    )
    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    print(f"Device: {device}; training frames: {len(train_set)}; validation frames: {len(validation_set)}")

    model = build_detector(config, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["training"]["epochs"])
    best_map = float("-inf")
    history: list[dict[str, float]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items() if key != "frame"} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running_loss += float(loss.detach())
        metrics = evaluate_map(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        history.append(record)
        print(record)
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch, "metrics": record}, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

    write_json(output_root / "training_history.json", {"device": str(device), "history": history, "best_map": best_map})
    print(f"Training complete. Best validation mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
