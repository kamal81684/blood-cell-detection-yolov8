"""Dataset statistics / EDA for the BCCD YOLO dataset."""
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from typing import Dict

from .utils import CLASS_NAMES


def dataset_stats(root: str) -> Dict[str, object]:
    """Compute per-split image counts and per-class box counts."""
    stats: Dict[str, object] = {}
    for split in ("train", "val", "test"):
        imgs = glob.glob(os.path.join(root, "images", split, "*"))
        lbls = glob.glob(os.path.join(root, "labels", split, "*.txt"))
        c = Counter()
        for lf in lbls:
            with open(lf) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        c[CLASS_NAMES[int(line.split()[0])]] += 1
        stats[split] = {"images": len(imgs), "boxes": dict(c),
                        "total_boxes": sum(c.values())}
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Print BCCD dataset statistics")
    p.add_argument("--root", required=True)
    args = p.parse_args()
    stats = dataset_stats(args.root)
    for split, s in stats.items():
        print(f"{split:5s} | images={s['images']:4d} | boxes={s['total_boxes']:5d} | {s['boxes']}")


if __name__ == "__main__":  # pragma: no cover
    main()
