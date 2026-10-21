"""Train a YOLOv8 detector on the prepared BCCD dataset.

Thin, reproducible wrapper around Ultralytics' trainer. All augmentation and
optimiser settings are passed explicitly so a run can be reproduced from the
command line / Makefile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import seed_everything


def train(data: str, epochs: int = 50, imgsz: int = 640, batch: int = 16,
          model: str = "yolov8n.pt", seed: int = 42, project: str = "runs/detect",
          name: str = "train", device: str = "cpu", resume: bool = False,
          workers: int = 0, cache: bool = False, plots: bool = True) -> str:
    """Train and return the path to the best weights."""
    from ultralytics import YOLO

    if device == "cpu":
        # Low-memory CPU mode: oneDNN needs a lot of scratch memory which is
        # scarce in small containers; single-thread keeps RSS predictable.
        import torch

        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(1)

    seed_everything(seed)
    net = YOLO(model)
    net.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        workers=workers,
        cache=cache,
        # augmentation / optimisation (explicit for reproducibility)
        optimizer="auto",
        cos_lr=True,
        patience=20,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0,
        val=True, plots=plots,
    )
    best = str(Path(project) / name / "weights" / "best.pt")
    print(f"best weights: {best}")
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv8 on BCCD")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default="train")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.model,
          args.seed, device=args.device, name=args.name, resume=args.resume,
          workers=args.workers, cache=args.cache, plots=not args.no_plots)


if __name__ == "__main__":  # pragma: no cover
    main()
