# Advanced Object Tracking and Detection in Video Streams

## 1. Problem and objective

The objective is to detect pedestrians in MOT17 video frames and preserve their identities over time. The proposed pipeline fine-tunes Faster R-CNN for pedestrian detection, then associates detections using an adaptive SORT tracker. The tracker contains a temporal consistency check: each identity's Kalman filter predicts its next box, and a new detection must agree with that prediction before the identity is continued.

## 2. Dataset and split

The supplied contains the MOT17 dataset. The implementation uses only `*-FRCNN` sequences when training because the `-DPM`, `-FRCNN`, and `-SDP` folders carry the same underlying video with different supplied detector outputs. This avoids accidental duplicate-frame training.

| Role | Sequences | Purpose |
|---|---|---|
| Training | MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-09-FRCNN | Fine-tune Faster R-CNN pedestrian detector |
| Validation/evaluation | MOT17-05-FRCNN | Report held-out detection and tracking metrics |

MOT17 ground-truth rows were filtered to valid pedestrian annotations (class 1, marked valid). Bounding boxes were changed from `(left, top, width, height)` to `(x1, y1, x2, y2)` for Faster R-CNN.

## 3. Preprocessing and augmentation

Frames are read as RGB, converted to floating-point tensors in `[0, 1]`, and passed to Torchvision Faster R-CNN. Its internal transform normalizes with ImageNet statistics and resizes each image between the configured 640-pixel minimum and 1280-pixel maximum while preserving aspect ratio. During training, gentle random crops, horizontal flips, brightness changes, contrast changes, and colour changes are applied. Bounding boxes are clipped/translated consistently with crops and flips. These operations improve robustness to illumination and viewpoint variation without changing identity labels.

## 4. Model and tracking method

The detector is Faster R-CNN with a ResNet-50 Feature Pyramid Network backbone, initialized from COCO weights and fine-tuned for background/pedestrian classes. The FPN supports small and large pedestrians at multiple image scales.

The tracking stage is an adaptive SORT implementation:

1. A constant-velocity Kalman filter predicts each box centre, width, and height.
2. Hungarian assignment matches predictions to current detections by IoU.
3. The temporal consistency gate accepts a match only if its IoU is above an adaptive threshold.
4. Faster tracks receive a modestly lower IoU gate and higher process uncertainty, accounting for larger inter-frame displacement. A minimum IoU prevents unreasonable matches.
5. Unmatched boxes create new identities; stale tracks are removed after `max_age` frames.

This design directly addresses dynamic scenes: prediction provides continuity through short detector misses, while adaptive gating makes association less brittle for fast-moving people.

## 5. Experimental configuration

Record the actual settings used before submission.

| Parameter | Value used |
|---|---|
| Epochs | `[from config / run]` |
| Batch size | `[from config / run]` |
| Image min/max size | `[from config / run]` |
| Detection score threshold | `[from config / run]` |
| Base/minimum IoU gate | `[from config / run]` |
| Maximum track age / minimum hits | `[from config / run]` |
| Hardware and runtime | `[GPU/CPU and approximate time]` |

## 6. Results

Paste exact figures from `results/detection_metrics.json` and `results/tracking_MOT17-05-FRCNN/tracking_metrics.json`. Do not round until the final table.

| Metric | Value | Interpretation |
|---|---:|---|
| mAP | `[actual]` | Overall detector precision across IoU thresholds |
| mAP@0.50 | `[actual]` | Detection performance with a 0.50 IoU match threshold |
| MOTA | `[actual]` | Tracking accuracy after false positives, misses, and switches |
| MOTP | `[actual]` | Localization precision of matched tracks |
| IDF1 | `[actual]` | Identity preservation quality across frames |
| Precision / Recall | `[actual] / [actual]` | Trade-off at the chosen detector threshold |
| Identity switches | `[actual]` | Count of incorrect ID changes; lower is better |

Include one screenshot or a short description of `annotated_tracking.mp4` here. Describe at least one successful case and one difficult case, such as occlusion, crowd density, motion blur, small people, or illumination change.

## 7. Analysis and justification

Use the sentences below as an editing guide, replacing bracketed portions with evidence from the actual run.

- The detector performed `[well/moderately/poorly]` on the held-out sequence because `[cite mAP, precision/recall, and observed scene characteristics]`.
- IDF1 was `[value]`, while the tracker made `[value]` identity switches. This indicates `[how consistently tracks kept the same person identity]`.
- If recall is low, likely explanations are a high confidence threshold, small/occluded pedestrians, insufficient fine-tuning, or a dataset split unlike the COCO pretraining distribution.
- If precision is low, likely explanations are crowded backgrounds, duplicate detections, a low confidence threshold, or overly relaxed association.
- The adaptive IoU gate helped when `[describe a fast-motion example]`; however, excessive relaxation can create false matches in dense crowds. The minimum IoU value mitigates this risk.

## 8. Limitations and future work

The tracker uses geometric motion and IoU, not deep appearance embeddings, so long occlusions and visually similar pedestrians can still cause identity switches. Improvements could include ReID embeddings (DeepSORT/BoT-SORT), camera-motion compensation, hyperparameter sweeps, a larger train split, and evaluation on more held-out MOT17 sequences. These extensions are optional; the submitted implementation already satisfies Faster R-CNN, temporal consistency, and adaptive tracking requirements.

## 9. Reproducibility

All paths and hyperparameters are in `config.yaml`. Run commands are in `README.md`, the code is under `src/`, and the notebook is included for output-based submission. The random seed is set to 42 for repeatable frame ordering and augmentation sampling, subject to normal GPU nondeterminism.
