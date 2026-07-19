"""Calculate tracking accuracy, IDF1, MOTA and identity switches on MOT17 train data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np

from src.common import load_config, resolve_path, write_json
from src.mot_dataset import read_ground_truth
from src.tracker import iou_matrix


def read_predictions(path: Path) -> dict[int, list[dict[str, float]]]:
    """Read standard 10-column MOT tracker results grouped by frame."""
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            frame, identity, left, top, width, height = map(float, row[:6])
            grouped[int(frame)].append({"id": int(identity), "left": left, "top": top, "width": width, "height": height})
    return dict(grouped)


def as_xyxy(items: list[dict[str, float]]) -> np.ndarray:
    """Convert MOT dictionaries to an Nx4 xyxy matrix."""
    return np.asarray([[item["left"], item["top"], item["left"] + item["width"], item["top"] + item["height"]] for item in items], dtype=float).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--predictions", required=True, help="tracking_results.txt from src/run_tracking.py")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    config, config_folder = load_config(args.config)
    sequence = args.sequence or config["data"]["validation_sequence"]
    root = resolve_path(config["paths"]["dataset_root"], config_folder)
    output_root = resolve_path(config["paths"]["output_root"], config_folder)
    ground_truth = read_ground_truth(root / "train" / sequence / "gt" / "gt.txt", float(config["data"]["min_visibility"]))
    predictions = read_predictions(Path(args.predictions))
    accumulator = mm.MOTAccumulator(auto_id=True)
    for frame in range(1, max(max(ground_truth, default=0), max(predictions, default=0)) + 1):
        truths = ground_truth.get(frame, [])
        estimates = predictions.get(frame, [])
        overlaps = iou_matrix(as_xyxy(truths), as_xyxy(estimates))
        distances = 1.0 - overlaps
        distances[overlaps < args.iou_threshold] = np.nan
        accumulator.update([item["id"] for item in truths], [item["id"] for item in estimates], distances)
    metrics = ["num_frames", "mota", "motp", "idf1", "precision", "recall", "num_switches", "mostly_tracked", "mostly_lost"]
    summary = mm.metrics.create().compute(accumulator, metrics=metrics, name=sequence)
    raw = summary.loc[sequence].to_dict()
    result = {key: (int(value) if key in {"num_frames", "num_switches", "mostly_tracked", "mostly_lost"} else float(value)) for key, value in raw.items()}
    result.update({"sequence": sequence, "iou_threshold": args.iou_threshold, "predictions": str(Path(args.predictions).resolve())})
    report_path = output_root / f"tracking_{sequence}" / "tracking_metrics.json"
    write_json(report_path, result)
    print(result)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
