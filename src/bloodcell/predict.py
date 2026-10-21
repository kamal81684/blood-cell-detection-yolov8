"""Run inference and count cells per class.

Used both as a CLI and by the Streamlit demo. Returns annotated images plus a
per-class count, which is the whole point of the project (a differential cell
count).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .preprocess import preprocess_image
from .utils import CLASS_NAMES


def _load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def count_cells(model, image: np.ndarray, conf: float = 0.25,
                preprocess: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
    """Detect cells in a BGR image; return (annotated BGR image, counts)."""
    import cv2

    src = preprocess_image(image) if preprocess else image
    res = model.predict(src, conf=conf, verbose=False)[0]
    annotated = res.plot()  # BGR numpy array with boxes/labels drawn

    counts = Counter()
    if res.boxes is not None and len(res.boxes) > 0:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for c in cls:
            counts[CLASS_NAMES[c]] += 1
    full = {n: int(counts.get(n, 0)) for n in CLASS_NAMES}
    return annotated, full


def predict_folder(weights: str, source: str, out: str = "runs/predict",
                   conf: float = 0.25, preprocess: bool = True) -> List[dict]:
    import cv2

    model = _load_model(weights)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for img_path in sorted(Path(source).glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path))
        annotated, counts = count_cells(model, img, conf, preprocess)
        cv2.imwrite(str(out_dir / f"pred_{img_path.name}"), annotated)
        records.append({"image": img_path.name, "counts": counts,
                        "total": sum(counts.values())})
        print(f"{img_path.name}: {counts}  (total={sum(counts.values())})")

    import json

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(records, fh, indent=2)
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Detect & count blood cells")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="image file or folder")
    p.add_argument("--out", default="runs/predict")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--no-preprocess", action="store_true")
    args = p.parse_args()
    pre = not args.no_preprocess

    src = Path(args.source)
    if src.is_dir():
        predict_folder(args.weights, args.source, args.out, args.conf, pre)
    else:
        import cv2

        model = _load_model(args.weights)
        img = cv2.imread(str(src))
        annotated, counts = count_cells(model, img, args.conf, pre)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(args.out) / f"pred_{src.name}"), annotated)
        print(f"{src.name}: {counts}  (total={sum(counts.values())})")


if __name__ == "__main__":  # pragma: no cover
    main()
