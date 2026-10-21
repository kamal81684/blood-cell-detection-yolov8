"""Assemble proof/evidence figures for the README from a training run.

Usage:
  python scripts/make_proofs.py \
      --run runs/detect/train \
      --preds runs/predict_test \
      --out assets
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["epoch"], df["train/box_loss"], "-o", label="train box")
    axes[0].plot(df["epoch"], df["train/cls_loss"], "-o", label="train cls")
    axes[0].plot(df["epoch"], df["train/dfl_loss"], "-o", label="train dfl")
    axes[0].plot(df["epoch"], df["val/box_loss"], "--s", label="val box")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Training / validation loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], "-o", label="mAP@0.5")
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], "-o", label="mAP@0.5:0.95")
    axes[1].plot(df["epoch"], df["metrics/precision(B)"], "--s", label="precision")
    axes[1].plot(df["epoch"], df["metrics/recall(B)"], "--s", label="recall")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].set_title("Validation metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("YOLOv8n on BCCD - training curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def grid_predictions(preds_dir: str, out_path: str, cols: int = 3) -> None:
    files = sorted(glob.glob(os.path.join(preds_dir, "*.jpg")))[:6]
    if not files:
        print("no predictions found in", preds_dir); return
    imgs = [cv2.imread(f) for f in files]
    imgs = [im for im in imgs if im is not None]
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, im, f in zip(axes, imgs, files):
        ax.imshow(im[:, :, ::-1])
        ax.set_title(os.path.basename(f), fontsize=9)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle("Predictions on held-out BCCD test images", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def plot_counts(counts: dict, out_path: str) -> None:
    names = ["RBC", "WBC", "Platelets"]
    imgs = list(counts.keys())
    x = np.arange(len(imgs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, n in enumerate(names):
        ax.bar(x + i * w, [counts[im].get(n, 0) for im in imgs], w, label=n)
    ax.set_xticks(x + w)
    ax.set_xticklabels([os.path.basename(i) for i in imgs], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Detected cells")
    ax.set_title("Per-image cell counts (held-out test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/detect/train")
    p.add_argument("--preds", default="runs/predict_test")
    p.add_argument("--out", default="assets")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    csv = os.path.join(args.run, "results.csv")
    if os.path.exists(csv):
        plot_training_curves(csv, os.path.join(args.out, "training_curves.png"))

    for src, dst in [(os.path.join(args.run, "BoxPR_curve.png"), "pr_curve.png"),
                     (os.path.join(args.run, "val_batch0_pred.jpg"), "val_predictions.jpg"),
                     (os.path.join(args.run, "val_batch0_labels.jpg"), "val_labels.jpg")]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, dst))
            print("copied", dst)

    grid_predictions(args.preds, os.path.join(args.out, "detections_grid.png"))

    # counts from the printed prediction summary is regenerated here
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from bloodcell.predict import count_cells
    from ultralytics import YOLO

    weights = os.path.join(args.run, "weights", "best.pt")
    if os.path.exists(weights):
        model = YOLO(weights)
        counts = {}
        for f in sorted(glob.glob(os.path.join(args.preds, "*.jpg")))[:6]:
            img = cv2.imread(f)
            _, c = count_cells(model, img)
            counts[f] = c
        plot_counts(counts, os.path.join(args.out, "cell_counts.png"))


if __name__ == "__main__":
    main()
