"""Convert the BCCD VOC-format annotations into Ultralytics YOLO layout.

Input  (BCCD repo):
    BCCD/JPEGImages/*.jpg
    BCCD/Annotations/*.xml          (Pascal VOC, classes: RBC / WBC / Platelets)
    BCCD/ImageSets/Main/{train,val,test}.txt   (file ids without extension)

Output (Ultralytics YOLO):
    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt          (cls cx cy w h, normalised)
    <out>/data.yaml

Optionally applies the OpenCV preprocessing pipeline while copying images
(--preprocess), which is the configuration used for the reported results.
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Tuple

import cv2

from .preprocess import load_image, preprocess_image
from .utils import CLASS_NAMES, ensure_dir, seed_everything

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_voc(xml_path: str) -> Tuple[Tuple[int, int], List[Tuple[int, float, float, float, float]]]:
    """Parse one VOC XML -> (image_size, [(class_id, cx, cy, w, h), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_TO_ID:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        # clamp to image bounds and convert to normalised YOLO cxcywh
        xmin, xmax = max(0.0, xmin), min(float(w), xmax)
        ymin, ymax = max(0.0, ymin), min(float(h), ymax)
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 1 or bh <= 1:
            continue
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        boxes.append((CLASS_TO_ID[name], cx, cy, bw / w, bh / h))
    return (w, h), boxes


def _read_split(imageset_dir: str, split: str) -> List[str]:
    path = os.path.join(imageset_dir, f"{split}.txt")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def prepare(src: str, out: str, preprocess: bool = False,
            seed: int = 42) -> Dict[str, Counter]:
    """Run the full conversion; returns per-split class counts."""
    seed_everything(seed)
    jpg_dir = os.path.join(src, "JPEGImages")
    ann_dir = os.path.join(src, "Annotations")
    sets_dir = os.path.join(src, "ImageSets", "Main")

    counts: Dict[str, Counter] = {}
    for split in ("train", "val", "test"):
        ids = _read_split(sets_dir, split)
        img_out = ensure_dir(os.path.join(out, "images", split))
        lbl_out = ensure_dir(os.path.join(out, "labels", split))
        c = Counter()
        for fid in ids:
            src_img = os.path.join(jpg_dir, fid + ".jpg")
            src_xml = os.path.join(ann_dir, fid + ".xml")
            if not (os.path.exists(src_img) and os.path.exists(src_xml)):
                continue
            img = load_image(src_img)
            if preprocess:
                img = preprocess_image(img)
            cv2.imwrite(os.path.join(img_out, fid + ".jpg"), img)

            _, boxes = parse_voc(src_xml)
            with open(os.path.join(lbl_out, fid + ".txt"), "w") as fh:
                for cid, cx, cy, bw, bh in boxes:
                    fh.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    c[CLASS_NAMES[cid]] += 1
        counts[split] = c
        print(f"[{split}] {len(ids)} images, class counts: {dict(c)}")

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {os.path.abspath(out)}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("test: images/test\n")
        fh.write(f"nc: {len(CLASS_NAMES)}\n")
        fh.write("names:\n")
        for i, n in enumerate(CLASS_NAMES):
            fh.write(f"  {i}: {n}\n")
    print(f"wrote {yaml_path}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="BCCD VOC -> YOLO conversion")
    p.add_argument("--src", required=True, help="path to BCCD_Dataset/BCCD")
    p.add_argument("--out", required=True, help="output YOLO dataset dir")
    p.add_argument("--preprocess", action="store_true",
                   help="apply OpenCV enhancement while copying")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare(args.src, args.out, args.preprocess, args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
