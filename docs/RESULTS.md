# Results & reproduction log

This file records exactly how the numbers and figures in the top-level `README.md`
were produced, so they can be reproduced.

## Environment

| | |
|---|---|
| OS | Linux (Debian 12), x86_64 |
| CPU | 2 vCPU, **2 GB RAM** (no GPU) |
| Python | 3.12 |
| Key deps | torch 2.14 (CPU), ultralytics 8.4, opencv 5.0, numpy 2.x |

The pod has only 2 GB of RAM and no GPU, so the committed run is a deliberately
small **smoke run**. Full training (50 epochs, 640 px) is the intended workflow
and is defined in `configs/bccd.yaml`.

## Commands

```bash
bash scripts/download_bccd.sh data
python -m bloodcell.prepare_data --src data/BCCD_Dataset-master/BCCD --out data/bccd --preprocess
python -m bloodcell.dataset --root data/bccd
# short CPU smoke run (batch size 2 to fit 2 GB RAM):
python -m bloodcell.train --data data/bccd/data.yaml --epochs 2 --imgsz 256 --batch 2 \
    --model yolov8n.pt --name smoke --workers 0 --no-plots
python -m bloodcell.predict --weights runs/detect/smoke/weights/best.pt \
    --source <test images> --out runs/predict_test --conf 0.25
```

## Dataset statistics (real BCCD)

```
train | images= 205 | boxes= 2804 | {'RBC': 2381, 'WBC': 214, 'Platelets': 209}
val   | images=  87 | boxes= 1137 | {'RBC': 967,  'WBC': 87,  'Platelets': 83}
test  | images=  72 | boxes=  945 | {'RBC': 805,  'WBC': 71,  'Platelets': 69}
```

## Training curves (smoke run, 2 epochs)

| epoch | box_loss | cls_loss | dfl_loss | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| 1 | 1.590 | 1.933 | 1.399 | 0.383 | 0.462 | 0.484 | 0.301 |
| 2 | 1.291 | 1.155 | 1.214 | 0.718 | 0.742 | 0.768 | 0.484 |

(Raw file: `runs/detect/smoke/results.csv`; plotted in `assets/training_curves.png`.)

## Validation metrics — per class

Produced by a clean `val` pass (`imgsz=192`, `batch=1`) with the smoke-run weights:

| Class | Images | Instances | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| all | 87 | 1137 | 0.695 | 0.651 | 0.655 | 0.413 |
| RBC | 83 | 967 | 0.661 | 0.785 | 0.766 | 0.459 |
| WBC | 86 | 87 | 0.900 | 1.000 | 0.967 | 0.687 |
| Platelets | 45 | 83 | 0.525 | 0.169 | 0.231 | 0.093 |

## Inference sanity check (held-out test images, conf 0.25)

| image | RBC | WBC | Platelets |
|---|---|---|---|
| BloodImage_00007 | 21 | 1 | 0 |
| BloodImage_00011 | 23 | 1 | 0 |
| BloodImage_00015 | 20 | 1 | 0 |
| BloodImage_00016 | 19 | 1 | 0 |
| BloodImage_00018 | 25 | 1 | 0 |
| BloodImage_00019 | 17 | 1 | 0 |

WBC is detected exactly once per image (correct — these smears each contain one
WBC), RBC counts are plausible, and platelets are missed by this under-trained
model, which is the expected failure mode.

## Figures → scripts

| Figure | Produced by |
|---|---|
| `assets/preprocessing_stages.png` | `scripts/make_figures.py` |
| `assets/sample_ground_truth.png` | `scripts/make_figures.py` |
| `assets/dataset_distribution.png` | `scripts/make_figures.py` |
| `assets/training_curves.png` | `scripts/make_proofs.py` |
| `assets/val_labels.jpg`, `assets/val_predictions.jpg` | Ultralytics val plots |
| `assets/detections_grid.png` | `scripts/make_proofs.py` |
| `assets/cell_counts.png` | `scripts/make_proofs.py` |
| `assets/per_class_metrics.png` | metrics table above |
