# Advanced_Object_Tracking_and_Detection_in_Video_Streams

This is a complete, reproducible submission package for **Advanced Object Tracking and Detection in Video Streams**. It fine-tunes a Faster R-CNN pedestrian detector on MOT17, checks detection-to-track temporal consistency with Kalman prediction, and uses an adaptive SORT tracker whose IoU association gate adjusts to motion speed.

The package does **not** include the 5.55 GiB uncompressed dataset. The assignment explicitly says data files need not be submitted; keep the supplied `archive (2).zip` on your computer and extract it locally.

## What is included

- `src/prepare_dataset.py` - safe one-time unpacking of the supplied ZIP.
- `src/train_detector.py` - Faster R-CNN fine-tuning and validation mAP.
- `src/run_tracking.py` - detector inference, temporal consistency, adaptive tracking, MOT text output, and annotated MP4.
- `src/evaluate_detector.py` and `src/evaluate_tracking.py` - mAP, MOTA, MOTP, IDF1, precision, recall, and identity-switch evaluation.
- `notebooks/Object_Tracking_and_Detection_in_Video_Streams.ipynb` - submission notebook to run top-to-bottom and export as HTML/PDF.
- `reports/REPORT_TEMPLATE.md` and `reports/VIDEO_SCRIPT.md` - a rubric-aligned report and an under-four-minute presentation plan.

## Exactly where to change paths

Open `config.yaml`. The only machine-specific values are at the top:

| Setting | Meaning | Change it when |
|---|---|---|
| `paths.dataset_zip` | Input archive used only by the extractor | The ZIP is in another folder |
| `paths.dataset_root` | Input folder containing `train/` and `test/` | You extracted MOT17 elsewhere |
| `paths.output_root` | All generated metrics, checkpoints, videos, and MOT text | You want results in another folder |
| `paths.checkpoint_path` | Output file for the best Faster R-CNN checkpoint | You rename/move the model file |

Use forward slashes in YAML even on Windows, for example `C:/Users/your_name/Documents/MOT17`.

## Run location and commands (Windows PowerShell)

1. Extract the assignment ZIP that you downloaded from this response. In PowerShell, change to the folder containing `config.yaml`:

```powershell
cd "C:\path\to\Adv_Object_Tracking_and_Detection_in_Video_Streams"
```

2. Create and activate a virtual environment (recommended), then install dependencies. Install the PyTorch command recommended for your CPU/CUDA version from [PyTorch's official installer](https://pytorch.org/get-started/locally/) before installing the remaining packages.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

3. Extract the supplied MOT17 archive once. This expands to about 5.55 GiB. The command below produces `C:\Users\Downloads\MOT17`, which matches the included `config.yaml`.

```powershell
python -m src.prepare_dataset --zip "C:\Users\Downloads\archive (2).zip" --destination "C:\Users\Downloads"
```

If your locations differ, edit `paths.dataset_zip` and `paths.dataset_root` in `config.yaml` now. Check that `dataset_root` ends in `MOT17` and contains folders named `train` and `test`.

```powershell
python scripts/check_setup.py --config config.yaml
```

4. Train and evaluate the detector. The first run downloads COCO pretrained Faster R-CNN weights; this is intentional transfer learning. Use a CUDA GPU if available. For a short installation check, add `--epochs 1`.

```powershell
python -m src.train_detector --config config.yaml
python -m src.evaluate_detector --config config.yaml
```

5. Produce a 300-frame qualitative tracking demo and its MOT-format text file. Change `runtime.max_frames_for_demo` or pass `--max-frames 0` to process the entire validation sequence.

```powershell
python -m src.run_tracking --config config.yaml
```

For the final values that go in the report, rerun tracking on the full validation sequence, then evaluate it:

```powershell
python -m src.run_tracking --config config.yaml --max-frames 0
python -m src.evaluate_tracking --config config.yaml --predictions ".\results\tracking_MOT17-05-FRCNN\tracking_results.txt"
```

## Expected outputs

After a successful run, the configured `paths.output_root` contains:

```text
results/
  checkpoints/fasterrcnn_best.pt       # best validation-mAP model
  training_history.json                # epoch loss and validation mAP
  detection_metrics.json               # COCO-style mAP/mAP@50/etc.
  tracking_MOT17-05-FRCNN/
    tracking_results.txt               # standard MOTChallenge predictions
    annotated_tracking.mp4             # qualitative video with IDs
    run_metadata.json                  # temporal/adaptive tracker parameters
    tracking_metrics.json              # MOTA, IDF1, ID switches, etc.
```

Do not place made-up scores in the report. Copy the values produced in `detection_metrics.json` and `tracking_metrics.json` into the results table in `reports/REPORT_TEMPLATE.md` after the run.

## Design choices

- **Preprocessing:** RGB conversion, image-tensor scaling, Faster R-CNN internal ImageNet normalization and min/max resizing, and pedestrian-only MOT17 annotations. Training augmentation includes gentle random cropping, horizontal flip, brightness, contrast, and colour variation.
- **Detector:** COCO-pretrained Faster R-CNN with a ResNet-50 FPN backbone is fine-tuned for two classes: background and pedestrian.
- **Temporal consistency:** a Kalman filter predicts each track location. A detector box is only linked to a track when its IoU with that predicted location clears a consistency gate.
- **Adaptive tracking:** the IoU gate relaxes modestly for high-speed tracks and Kalman process noise grows with speed. This reduces missed associations caused by rapid motion while preserving a lower safety bound.
- **Evaluation:** mAP describes detector localization/classification; MOTA, IDF1, precision/recall, and identity switches describe tracking quality. Use the same held-out `MOT17-05-FRCNN` sequence for the recorded values.

## Baseline comparison

MOT17 includes supplied FRCNN detections. To isolate the tracker and compare this baseline against the fine-tuned detector, run:

```powershell
python -m src.run_tracking --config config.yaml --detections-file "C:\Users\Downloads\MOT17\train\MOT17-05-FRCNN\det\det.txt" --no-video
```

Evaluate that run, rename or save its generated JSON, then repeat with the fine-tuned model. Clearly label the supplied-detection run as a baseline in the report.

## Common issues

- `ModuleNotFoundError`: activate `.venv` and rerun `pip install -r requirements.txt`.
- `CUDA out of memory`: set `training.batch_size: 1`, `model.min_size: 512`, and `model.max_size: 1024` in `config.yaml`.
- Missing checkpoint: complete `src.train_detector.py` before `src.run_tracking.py`.
- Windows multiprocessing errors: retain `training.num_workers: 0`.
- Slow full sequence: keep the 300-frame demo for the presentation; run `--max-frames 0` when collecting final tracking metrics.
