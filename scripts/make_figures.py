"""Generate documentation figures (proofs) from the real BCCD dataset.

Outputs into assets/:
  - preprocessing_stages.png   : original -> CLAHE -> denoise -> HSV norm
  - dataset_distribution.png   : per-split class counts
  - sample_ground_truth.png    : a BCCD image with its ground-truth boxes

Usage:  python scripts/make_figures.py --bccd data/BCCD_Dataset-master/BCCD --out assets
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bloodcell.preprocess import stage_visualisation  # noqa: E402
from bloodcell.utils import CLASS_NAMES  # noqa: E402

COLORS = {"RBC": (60, 60, 235), "WBC": (235, 120, 20), "Platelets": (40, 200, 40)}


def fig_preprocessing(img_bgr: np.ndarray, out_path: str) -> None:
    stages = stage_visualisation(img_bgr)
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4))
    for ax, (name, im) in zip(axes, stages.items()):
        ax.imshow(im[:, :, ::-1])
        ax.set_title(name.replace("_", " ").title(), fontsize=12)
        ax.axis("off")
    fig.suptitle("OpenCV preprocessing pipeline (real BCCD sample)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def fig_distribution(bccd_root: str, out_path: str) -> None:
    ann_dir = os.path.join(bccd_root, "Annotations")
    sets_dir = os.path.join(bccd_root, "ImageSets", "Main")
    data = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(sets_dir, f"{split}.txt")) as fh:
            ids = [ln.strip() for ln in fh if ln.strip()]
        c = Counter()
        for fid in ids:
            xml = os.path.join(ann_dir, fid + ".xml")
            if not os.path.exists(xml):
                continue
            for obj in ET.parse(xml).getroot().findall("object"):
                name = obj.find("name").text.strip()
                if name in CLASS_NAMES:
                    c[name] += 1
        data[split] = [c.get(n, 0) for n in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, split in enumerate(("train", "val", "test")):
        ax.bar(x + i * w, data[split], w, label=split)
    ax.set_xticks(x + w)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Number of annotated cells")
    ax.set_title("BCCD class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def fig_ground_truth(img_bgr: np.ndarray, xml_path: str, out_path: str) -> None:
    root = ET.parse(xml_path).getroot()
    vis = img_bgr.copy()
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in COLORS:
            continue
        bb = obj.find("bndbox")
        x1, y1 = int(float(bb.find("xmin").text)), int(float(bb.find("ymin").text))
        x2, y2 = int(float(bb.find("xmax").text)), int(float(bb.find("ymax").text))
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[name], 2)
        cv2.putText(vis, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLORS[name], 2)
    cv2.imwrite(out_path, vis)
    print("wrote", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bccd", default="data/BCCD_Dataset-master/BCCD")
    p.add_argument("--out", default="assets")
    p.add_argument("--sample", default="BloodImage_00007")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = cv2.imread(os.path.join(args.bccd, "JPEGImages", args.sample + ".jpg"))
    fig_preprocessing(img, os.path.join(args.out, "preprocessing_stages.png"))
    fig_distribution(args.bccd, os.path.join(args.out, "dataset_distribution.png"))
    fig_ground_truth(img, os.path.join(args.bccd, "Annotations", args.sample + ".xml"),
                     os.path.join(args.out, "sample_ground_truth.png"))


if __name__ == "__main__":
    main()
