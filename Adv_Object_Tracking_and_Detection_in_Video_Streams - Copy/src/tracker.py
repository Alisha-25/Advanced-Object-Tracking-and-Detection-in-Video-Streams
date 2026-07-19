"""Adaptive SORT-style tracker with temporal consistency gating.

Each track predicts its next bounding box through a constant-velocity Kalman
filter.  Matching uses a speed-adaptive IoU gate: faster tracks are allowed a
slightly lower overlap because frame-to-frame displacement is larger.  This is
the temporal consistency and adaptive tracking component required by the brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for arrays of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def xyxy_to_measurement(box: np.ndarray) -> np.ndarray:
    """Convert a box to [centre_x, centre_y, width, height]."""
    x1, y1, x2, y2 = box.astype(float)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def measurement_to_xyxy(measurement: np.ndarray) -> np.ndarray:
    """Convert [centre_x, centre_y, width, height] back to a valid xyxy box."""
    cx, cy, width, height = measurement.astype(float)
    width, height = max(width, 1.0), max(height, 1.0)
    return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], dtype=float)


@dataclass
class KalmanTrack:
    """A constant-velocity box filter and its identity bookkeeping."""

    track_id: int
    initial_box: np.ndarray
    score: float
    process_noise: float
    measurement_noise: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    x: np.ndarray = field(init=False)
    p: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x = np.zeros(8, dtype=float)
        self.x[:4] = xyxy_to_measurement(self.initial_box)
        self.p = np.diag([20, 20, 20, 20, 100, 100, 100, 100]).astype(float)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[4:6]))

    def predict(self) -> np.ndarray:
        """Propagate a track one frame, increasing uncertainty for faster motion."""
        transition = np.eye(8)
        transition[0, 4] = transition[1, 5] = transition[2, 6] = transition[3, 7] = 1.0
        scale = 1.0 + self.speed / 50.0
        process = np.diag([1, 1, 2, 2, 8, 8, 4, 4]).astype(float) * self.process_noise * scale
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process
        self.age += 1
        self.time_since_update += 1
        return measurement_to_xyxy(self.x[:4])

    def update(self, box: np.ndarray, score: float) -> None:
        """Correct a predicted track with the assigned detection."""
        observation = xyxy_to_measurement(box)
        measurement_matrix = np.zeros((4, 8))
        measurement_matrix[:, :4] = np.eye(4)
        noise = np.eye(4) * self.measurement_noise
        innovation = observation - measurement_matrix @ self.x
        covariance = measurement_matrix @ self.p @ measurement_matrix.T + noise
        gain = self.p @ measurement_matrix.T @ np.linalg.inv(covariance)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(8) - gain @ measurement_matrix) @ self.p
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0

    def box(self) -> np.ndarray:
        return measurement_to_xyxy(self.x[:4])


class AdaptiveSORT:
    """A detector-agnostic tracker with adaptive temporal association rules."""

    def __init__(self, config: dict) -> None:
        self.base_iou = float(config["base_iou_threshold"])
        self.min_iou = float(config["min_iou_threshold"])
        self.velocity_relaxation = float(config["velocity_relaxation"])
        self.max_age = int(config["max_age"])
        self.min_hits = int(config["min_hits"])
        self.process_noise = float(config["process_noise"])
        self.measurement_noise = float(config["measurement_noise"])
        self.tracks: list[KalmanTrack] = []
        self.next_id = 1
        self.frame_count = 0

    def _adaptive_threshold(self, track: KalmanTrack) -> float:
        # High speed lowers the gate by at most velocity_relaxation, avoiding
        # spurious missed matches under fast motion while retaining a minimum.
        return max(self.min_iou, self.base_iou - min(track.speed / 100.0, 1.0) * self.velocity_relaxation)

    def update(self, detections: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
        """Associate detections and return confirmed tracks for the current frame.

        Args:
            detections: Nx4 xyxy boxes after detector score thresholding.
            scores: N detector confidences aligned with ``detections``.
        """
        detections = np.asarray(detections, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        self.frame_count += 1
        predictions = np.array([track.predict() for track in self.tracks], dtype=float).reshape(-1, 4)
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        if len(self.tracks) and len(detections):
            overlaps = iou_matrix(predictions, detections)
            track_indices, detection_indices = linear_sum_assignment(-overlaps)
            for track_index, detection_index in zip(track_indices, detection_indices):
                # Temporal consistency check: a detection must agree with the
                # Kalman prediction under the track's speed-adaptive gate.
                if overlaps[track_index, detection_index] >= self._adaptive_threshold(self.tracks[track_index]):
                    matches.append((track_index, detection_index))
                    unmatched_tracks.discard(track_index)
                    unmatched_detections.discard(detection_index)
        for track_index, detection_index in matches:
            self.tracks[track_index].update(detections[detection_index], scores[detection_index])
        for detection_index in sorted(unmatched_detections):
            self.tracks.append(KalmanTrack(self.next_id, detections[detection_index], scores[detection_index], self.process_noise, self.measurement_noise))
            self.next_id += 1
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        return [
            {"track_id": track.track_id, "box": track.box(), "score": track.score, "speed": track.speed}
            for track in self.tracks
            if track.time_since_update == 0 and (track.hits >= self.min_hits or self.frame_count <= self.min_hits)
        ]
