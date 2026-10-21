"""Unit tests for the preprocessing pipeline and VOC->YOLO conversion."""
import numpy as np
import pytest

from bloodcell.prepare_data import parse_voc
from bloodcell.preprocess import (PreprocessConfig, apply_clahe, denoise,
                                  hsv_normalize, preprocess_image)


@pytest.fixture()
def dummy_bgr():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_preprocess_preserves_shape_and_dtype(dummy_bgr):
    out = preprocess_image(dummy_bgr)
    assert out.shape == dummy_bgr.shape
    assert out.dtype == np.uint8


def test_clahe_preserves_shape(dummy_bgr):
    assert apply_clahe(dummy_bgr).shape == dummy_bgr.shape


def test_denoise_off_is_identity(dummy_bgr):
    cfg = PreprocessConfig(use_denoise=False)
    assert np.array_equal(denoise(dummy_bgr, cfg), dummy_bgr)


def test_hsv_normalize_range(dummy_bgr):
    out = hsv_normalize(dummy_bgr, PreprocessConfig())
    assert out.min() >= 0 and out.max() <= 255


def test_preprocess_is_deterministic(dummy_bgr):
    a = preprocess_image(dummy_bgr)
    b = preprocess_image(dummy_bgr)
    assert np.array_equal(a, b)


def test_parse_voc(tmp_path):
    xml = """<annotation><size><width>100</width><height>50</height><depth>3</depth></size>
    <object><name>WBC</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>
    </annotation>"""
    p = tmp_path / "a.xml"
    p.write_text(xml)
    (w, h), boxes = parse_voc(str(p))
    assert (w, h) == (100, 50)
    assert len(boxes) == 1
    cid, cx, cy, bw, bh = boxes[0]
    assert cid == 1  # WBC
    assert 0 < cx < 1 and 0 < cy < 1 and 0 < bw < 1 and 0 < bh < 1
