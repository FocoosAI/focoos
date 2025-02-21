import base64
import math

import numpy as np
import supervision as sv

from focoos.ports import FocoosDet, FocoosDetections
from focoos.utils.vision import (
    base64mask_to_mask,
    binary_mask_to_base64,
    class_to_index,
    fai_detections_to_sv,
    image_loader,
    image_preprocess,
    index_to_class,
    scale_detections,
    scale_mask,
    sv_to_fai_detections,
)


def test_index_to_class():
    class_ids = [0, 1, 2]
    classes = ["cat", "dog", "bird"]
    assert index_to_class(class_ids, classes) == ["cat", "dog", "bird"]


def test_class_to_index():
    classes = ["cat", "dog", "bird"]
    class_names = ["cat", "dog", "bird"]
    assert class_to_index(classes, class_names) == [0, 1, 2]


def test_image_loader_pil_image(pil_image):
    image = image_loader(pil_image)
    assert isinstance(image, np.ndarray)
    assert image.shape == (640, 640, 3)


def test_image_loader_image_bytes(image_bytes):
    image = image_loader(image_bytes)
    assert isinstance(image, np.ndarray)
    assert image.shape == (640, 640, 3)


def test_image_loader_image_path(image_path):
    image = image_loader(image_path)
    assert isinstance(image, np.ndarray)
    assert image.shape == (640, 640, 3)


def test_image_loader_image_ndarray(image_ndarray):
    image = image_loader(image_ndarray)
    assert isinstance(image, np.ndarray)
    assert image.shape == (640, 640, 3)


def test_image_preprocess_resize(pil_image):
    resize_dim = 100

    # Call the function with resize
    im1, im0 = image_preprocess(pil_image, resize=resize_dim)

    # Ensure the resized image shape matches (100, 100, 3)
    assert im0.shape == (
        pil_image.height,
        pil_image.width,
        3,
    ), f"Expected shape {(pil_image.height, pil_image.width, 3)}, but got {im0.shape}"

    # Ensure that im1 has shape (1, 3, 100, 100) after processing
    assert im1.shape == (
        1,
        3,
        resize_dim,
        resize_dim,
    ), f"Expected shape (1, 3, {resize_dim}, {resize_dim}), but got {im1.shape}"


def test_scale_mask():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    scaled = scale_mask(mask, (4, 4))
    assert scaled.shape == (4, 4)


def test_scale_detections_no_scaling_needed():
    detections = sv.Detections(xyxy=np.array([[10, 20, 30, 40]]))
    result = scale_detections(detections, (100, 100), (100, 100))
    np.testing.assert_array_equal(result.xyxy, np.array([[10, 20, 30, 40]]))


def test_scale_detections_scaling_applied():
    detections = sv.Detections(xyxy=np.array([[10, 20, 30, 40]]))
    in_shape = (100, 100)
    out_shape = (200, 200)
    result = scale_detections(detections, in_shape, out_shape)
    expected_xyxy = np.array([[20, 40, 60, 80]])  # Expected scaled values
    np.testing.assert_array_equal(result.xyxy, expected_xyxy)


def test_base64mask_to_mask(image_bytes):
    base64ask = base64.b64encode(image_bytes).decode("utf-8")

    result = base64mask_to_mask(base64ask)

    # Verify the result is a NumPy array
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    # Verify the shape matches the original image
    assert result.shape == (640, 640, 3), "Decoded image shape is incorrect"


def test_focoos_detections_to_supervision_bbox(focoos_detections_bbox):
    result = fai_detections_to_sv(focoos_detections_bbox)

    # Verify the result is an instance of Supervision Detections
    assert isinstance(result[0], sv.Detections), "Result should be an instance of Supervision Detections"
    # Verify the number of detections
    assert len(result.xyxy) == 1, "Expected 1 detection"
    # Verify the bounding box coordinates
    np.testing.assert_array_equal(result.xyxy, np.array([[0, 0, 1, 1]]))
    # Verify the class ID
    assert result.class_id == [1], "Expected class ID 1"
    # Verify the confidence score
    assert result.confidence == [0.9], "Expected confidence score 0.9"


def test_focoos_detections_to_supervision_mask(focoos_detections_mask):
    result = fai_detections_to_sv(focoos_detections_mask)

    # Verify the result is an instance of Supervision Detections
    assert isinstance(result[0], sv.Detections), "Result should be an instance of Supervision Detections"
    # # Verify the number of detections
    # FIXME: https://github.com/FocoosAI/focoos/issues/38
    # assert len(result.xyxy) == 0, "Expected 0 detection"
    # Verify the mask
    assert isinstance(result.mask, np.ndarray), "Mask should be a NumPy array"
    assert result.mask.shape == (1, 2, 2), "Mask shape is incorrect"


def test_focoos_detections_no_detections(focoos_detections_no_detections):
    result = fai_detections_to_sv(focoos_detections_no_detections)

    # Verify the result is an instance of Supervision Detections
    assert isinstance(result, sv.Detections), "Result should be an instance of sv.Detections"
    # Verify the number of detections
    assert len(result.xyxy) == 0, "Expected 0 detection"
    # Verify the mask is None
    assert result.mask is None, "Mask should be None"


def test_binary_mask_to_base64(binary_mask, base64_binary_mask):
    result = binary_mask_to_base64(binary_mask)

    # Verify the result is a string
    assert isinstance(result, str), "Result should be a string"
    assert result == base64_binary_mask, "Base64-encoded mask is incorrect"


def test_sv_to_focoos_detections(sv_detections: sv.Detections):
    result = sv_to_fai_detections(sv_detections)

    # Verify the result is an instance of FocoosDetections
    assert isinstance(result, FocoosDetections), "Result should be an instance of FocoosDetections"
    assert len(result.detections) == 1, "Expected 1 detection"
    result_focoos_detection = result.detections[0]
    # Verify the result is an instance of FocoosDet
    assert isinstance(result_focoos_detection, FocoosDet), "Result should be an instance of FocoosDet"

    assert result_focoos_detection.cls_id == 1, "Expected class ID 1"
    assert result_focoos_detection.label is None, "Label should be None"
    assert result_focoos_detection.conf is not None, "Confidence score should not be None"
    assert math.isclose(result_focoos_detection.conf, 0.9), "Expected confidence score 0.9"
    assert result_focoos_detection.bbox == [
        10,
        20,
        30,
        40,
    ], "Bounding box coordinates are incorrect"
    assert isinstance(result_focoos_detection.mask, str), "Mask should be a string"
