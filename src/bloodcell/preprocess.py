"""OpenCV preprocessing pipeline for blood-smear microscopy images.

The BCCD images are stained peripheral-blood smears. Staining intensity and
microscope illumination vary between images, which hurts a detector's ability
to separate faint cells (especially platelets) from the background. This module
applies a small, deterministic enhancement chain:

    1. CLAHE contrast enhancement on the L channel of LAB space
    2. Gaussian / non-local-means denoising
    3. HSV-based colour normalisation (V rescaled, S clipped)

All steps are pure OpenCV + NumPy, so they run on CPU in a few milliseconds and
are trivially reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Knobs for the enhancement pipeline (sensible defaults for BCCD)."""

    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    denoise_h: float = 7.0
    denoise_h_color: float = 7.0
    denoise_template: int = 7
    denoise_search: int = 21
    hsv_s_max: float = 255.0
    use_denoise: bool = True
    # "bilateral" (fast, edge-preserving) or "nlm" (slow, higher quality)
    denoise_method: str = "bilateral"
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a single-channel grayscale copy of a BGR/RGB image."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the LAB L channel.

    CLAHE is applied to luminance only so hue/saturation (which carry the
    stain colour) are preserved.
    """
    if img.ndim == 2:
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Edge-preserving denoising to suppress sensor/microscope grain."""
    if not cfg.use_denoise:
        return img
    if cfg.denoise_method == "bilateral":
        return cv2.bilateralFilter(img, cfg.bilateral_d,
                                   cfg.bilateral_sigma_color,
                                   cfg.bilateral_sigma_space)
    # non-local means: slower but higher quality
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, cfg.denoise_h,
                                        cfg.denoise_template, cfg.denoise_search)
    return cv2.fastNlMeansDenoisingColored(img, None, cfg.denoise_h,
                                           cfg.denoise_h_color,
                                           cfg.denoise_template, cfg.denoise_search)


def hsv_normalize(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Rescale value (brightness) and clip saturation to reduce stain drift."""
    if img.ndim == 2:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v = (v - v.min()) / (float(np.ptp(v)) + 1e-6) * 255.0
    s = np.clip(s, 0, cfg.hsv_s_max)
    out = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def preprocess_image(img: np.ndarray, cfg: Optional[PreprocessConfig] = None) -> np.ndarray:
    """Run the full enhancement chain on a BGR image and return a BGR image."""
    cfg = cfg or PreprocessConfig()
    out = apply_clahe(img, cfg.clahe_clip, cfg.clahe_grid)
    out = denoise(out, cfg)
    out = hsv_normalize(out, cfg)
    return out


def load_image(path: str) -> np.ndarray:
    """Read an image from disk as BGR uint8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def stage_visualisation(img: np.ndarray, cfg: Optional[PreprocessConfig] = None):
    """Return the intermediate stages as an ordered dict (for docs/plots)."""
    cfg = cfg or PreprocessConfig()
    a = apply_clahe(img, cfg.clahe_clip, cfg.clahe_grid)
    b = denoise(a, cfg)
    c = hsv_normalize(b, cfg)
    return {"original": img, "clahe": a, "denoise": b, "hsv_norm": c}


def _main() -> None:  # pragma: no cover - tiny CLI
    import argparse
    p = argparse.ArgumentParser(description="Preprocess a blood-smear image")
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    cv2.imwrite(args.output, preprocess_image(load_image(args.input)))
    print(f"wrote {args.output}")


if __name__ == "__main__":  # pragma: no cover
    _main()
