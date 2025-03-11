import base64

import cv2
import numpy as np
import pytest
import supervision as sv

from focoos.ports import FocoosDet, FocoosDetections


@pytest.fixture
def binary_mask() -> np.ndarray:
    """Create a simple binary mask."""
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    return mask


@pytest.fixture
def base64_binary_mask(binary_mask: np.ndarray) -> str:
    """Create a base64-encoded binary mask using OpenCV."""
    mask = binary_mask.astype(np.uint8) * 255
    success, encoded_image = cv2.imencode(".png", mask)
    if not success:
        raise ValueError("Failed to encode image")
    encoded_mask = base64.b64encode(encoded_image).decode("utf-8")
    return encoded_mask


@pytest.fixture
def focoos_detections_bbox() -> FocoosDetections:
    """Create a sample FocoosDetections object with bbox."""
    detections = FocoosDetections(
        detections=[
            FocoosDet(
                bbox=[0, 0, 1, 1],
                cls_id=1,
                conf=0.9,
            ),
        ]
    )
    return detections


@pytest.fixture
def focoos_detections_mask(base64_binary_mask) -> FocoosDetections:
    """Create a sample FocoosDetections object with mask."""
    detections = FocoosDetections(
        detections=[
            FocoosDet(
                bbox=[0, 0, 1, 1],
                cls_id=1,
                conf=0.9,
                mask=base64_binary_mask,
            )
        ]
    )
    return detections


@pytest.fixture
def focoos_detections_no_detections() -> FocoosDetections:
    """Create a sample FocoosDetections object with no detections."""
    detections = FocoosDetections(detections=[])
    return detections


@pytest.fixture
def sv_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
        class_id=np.array([1, 2, 3]),
        confidence=np.array([0.9, 0.8, 0.7]),
        mask=np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=bool),
    )
