"""bloodcell: blood cell detection & counting with YOLOv8 + OpenCV preprocessing.

Public API
----------
- preprocess.preprocess_image : OpenCV enhancement pipeline
- prepare_data.prepare        : BCCD VOC -> YOLO conversion
- predict.count_cells         : run a trained detector and count cells by class
"""

__version__ = "0.1.0"

from .preprocess import preprocess_image  # noqa: F401

__all__ = ["preprocess_image", "__version__"]
