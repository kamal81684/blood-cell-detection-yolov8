"""Small shared helpers: seeding, paths, and plotting."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy (and torch if installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional at import time
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


CLASS_NAMES = ["RBC", "WBC", "Platelets"]


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1].copy()
