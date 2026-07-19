"""Run fine-tuned Faster R-CNN plus AdaptiveSORT and export a labelled MP4/MOT file."""

from __future__ import annotations

import argparse
import configparser
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.common import get_device, load_config, resolve_path, write_json
from src.detector import load_detector
from src.tracker import AdaptiveSORT


def load_mot_detections(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read existing MOT detector boxes for an optional tracking baseline."""
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 7:
                grouped[int(float(row[0]))].append([float(value) for value in row[2:7]])
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame, rows in grouped.items():
        values = np.asarray(rows, dtype=float)
        boxes = values[:, :4].copy()  # left, top, width, height
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        output[frame] = boxes, values[:, 4]
    return output


@torch.inference_mode()
def detector_boxes(
    model: torch.nn.Module, image_path: Path, device: torch.device, score_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Predict person boxes from one frame using fine-tuned Faster R-CNN."""
    image = TF.to_tensor(Image.open(image_path).convert("RGB")).to(device)
    output = model([image])[0]
    keep = (output["labels"] == 1) & (output["scores"] >= score_threshold)
    return output["boxes"][keep].detach().cpu().numpy(), output["scores"][keep].detach().cpu().numpy()


def draw_tracks(frame: np.ndarray, tracks: list[dict[str, float]]) -> np.ndarray:
    """Overlay identities and adaptive motion speed on an OpenCV BGR frame."""
    for track in tracks:
        x1, y1, x2, y2 = np.rint(track["box"]).astype(int)
        colour = (37 * int(track["track_id"]) % 255, 17 * int(track["track_id"]) % 255, 29 * int(track["track_id"]) % 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"ID {track['track_id']} | {track['score']:.2f} | v={track['speed']:.1f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 2, cv2.LINE_AA)
    return frame


def sequence_fps(sequence_dir: Path) -> float:
    """Read FPS from the MOT seqinfo file, defaulting safely to 30."""
    parser = configparser.ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini")
    return parser.getfloat("Sequence", "frameRate", fallback=30.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sequence", default=None, help="A train sequence, e.g. MOT17-05-FRCNN.")
    parser.add_argument("--max-frames", type=int, default=None, help="0 means process the entire sequence.")
    parser.add_argument("--detections-file", default=None, help="Optional MOT det.txt baseline; skips Faster R-CNN inference.")
    parser.add_argument("--no-video", action="store_true", help="Write only MOT text and metadata.")
    args = parser.parse_args()

    config, config_folder = load_config(args.config)
    root = resolve_path(config["paths"]["dataset_root"], config_folder)
    output_root = resolve_path(config["paths"]["output_root"], config_folder)
    sequence = args.sequence or config["data"]["validation_sequence"]
    sequence_dir = root / "train" / sequence
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No frames found in {sequence_dir / 'img1'}")
    max_frames = config["runtime"]["max_frames_for_demo"] if args.max_frames is None else args.max_frames
    if max_frames and max_frames > 0:
        image_paths = image_paths[:max_frames]

    tracking_cfg = config["tracking"]
    detector_baseline = Path(args.detections_file) if args.detections_file else None
    device = get_device(config["runtime"]["device"])
    model = None if detector_baseline else load_detector(resolve_path(config["paths"]["checkpoint_path"], config_folder), config, device)
    baseline_detections = load_mot_detections(detector_baseline) if detector_baseline else {}
    tracker = AdaptiveSORT(tracking_cfg)
    run_dir = output_root / f"tracking_{sequence}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "tracking_results.txt"
    video_writer: cv2.VideoWriter | None = None
    if not args.no_video:
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"Could not read {image_paths[0]}")
        height, width = first.shape[:2]
        video_writer = cv2.VideoWriter(str(run_dir / "annotated_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sequence_fps(sequence_dir), (width, height))

    rows_written = 0
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, image_path in enumerate(image_paths, start=1):
            frame_number = int(image_path.stem)
            if detector_baseline:
                boxes, scores = baseline_detections.get(frame_number, (np.empty((0, 4)), np.empty((0,))))
                keep = scores >= float(tracking_cfg["detection_score_threshold"])
                boxes, scores = boxes[keep], scores[keep]
            else:
                boxes, scores = detector_boxes(model, image_path, device, float(tracking_cfg["detection_score_threshold"]))
            tracks = tracker.update(boxes, scores)
            for track in tracks:
                x1, y1, x2, y2 = track["box"]
                writer.writerow([frame_number, track["track_id"], x1, y1, x2 - x1, y2 - y1, track["score"], -1, -1, -1])
                rows_written += 1
            if video_writer:
                frame = cv2.imread(str(image_path))
                video_writer.write(draw_tracks(frame, tracks))
            if index % 50 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)} frames")
    if video_writer:
        video_writer.release()
    write_json(run_dir / "run_metadata.json", {
        "sequence": sequence,
        "frames_processed": len(image_paths),
        "mot_rows_written": rows_written,
        "detection_source": str(detector_baseline) if detector_baseline else "fine-tuned Faster R-CNN",
        "temporal_consistency": "Kalman prediction + speed-adaptive IoU association",
        "tracking_parameters": tracking_cfg,
    })
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
