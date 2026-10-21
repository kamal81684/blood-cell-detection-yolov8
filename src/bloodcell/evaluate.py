"""Evaluate a trained YOLOv8 detector on BCCD: mAP, per-class metrics, confusion matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def evaluate(weights: str, data: str, imgsz: int = 640, split: str = "val",
             out: str = "runs/eval") -> Dict[str, float]:
    """Run validation and return a dict of headline metrics."""
    from ultralytics import YOLO

    Path(out).mkdir(parents=True, exist_ok=True)
    net = YOLO(weights)
    metrics = net.val(data=data, imgsz=imgsz, split=split, project=out,
                      name="val", exist_ok=True, plots=True)

    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_map50": {n: float(v) for n, v in
                            zip(metrics.names.values(), metrics.box.maps[: len(metrics.names)])},
    }
    with open(Path(out) / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 on BCCD")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default="runs/eval")
    args = p.parse_args()
    evaluate(args.weights, args.data, args.imgsz, args.split, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
