import base64
import math

import numpy as np
import supervision as sv

from focoos.ports import FocoosDet, FocoosTask
from focoos.utils.vision import (
    base64mask_to_mask,
    binary_mask_to_base64,
    class_to_index,
    det_postprocess,
    fai_detections_to_sv,
    get_postprocess_fn,
    image_loader,
    image_preprocess,
    index_to_class,
    instance_postprocess,
    masks_to_xyxy,
    scale_detections,
    scale_mask,
    semseg_postprocess,
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
    print(f"RESULT SHAPE {result.shape}")
    # Verify the result is a NumPy array
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    # Verify the shape matches the original image
    assert result.shape == (640, 640), "Decoded image shape is incorrect"


def test_focoos_detections_to_supervision_bbox(focoos_detections_bbox):
    result = fai_detections_to_sv(focoos_detections_bbox, im0_shape=(640, 640))

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
    result = fai_detections_to_sv(focoos_detections_mask, im0_shape=(2, 2))

    # Verify the result is an instance of Supervision Detections
    assert isinstance(result[0], sv.Detections), "Result should be an instance of Supervision Detections"
    # # Verify the number of detections
    # FIXME: https://github.com/FocoosAI/focoos/issues/38
    # assert len(result.xyxy) == 0, "Expected 0 detection"
    # Verify the mask
    assert isinstance(result.mask, np.ndarray), "Mask should be a NumPy array"
    assert result.mask.shape == (1, 2, 2), "Mask shape is incorrect"


def test_focoos_detections_no_detections(focoos_detections_no_detections):
    result = fai_detections_to_sv(focoos_detections_no_detections, im0_shape=(640, 640))

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
    assert all(isinstance(det, FocoosDet) for det in result), "All elements in result should be instances of FocoosDet"
    assert len(result) == 3, "Expected 3 detection"
    result_focoos_detection = result[0]
    # Verify the result is an instance of FocoosDet
    assert isinstance(result_focoos_detection, FocoosDet), "Result should be an instance of FocoosDet"

    assert result_focoos_detection.cls_id == 1, "Expected class ID 1"
    assert result_focoos_detection.label is None, "Label should be None"
    assert result_focoos_detection.conf is not None, "Confidence score should not be None"
    assert math.isclose(result_focoos_detection.conf, 0.9), "Expected confidence score 0.9"
    assert result_focoos_detection.bbox == [
        0,
        0,
        1,
        1,
    ], "Bounding box coordinates are incorrect"
    assert isinstance(result_focoos_detection.mask, str), "Mask should be a string"


def test_masks_to_xyxy():
    # Basic case: a single mask with one active pixel
    mask1 = np.zeros((1, 5, 5), dtype=bool)
    mask1[0, 2, 3] = True  # One active pixel at (2,3)
    assert np.array_equal(masks_to_xyxy(mask1), np.array([[3, 2, 3, 2]]))

    # Case with a rectangle
    mask2 = np.zeros((1, 5, 5), dtype=bool)
    mask2[0, 1:4, 2:5] = True  # Rectangle between (1,2) and (3,4)
    assert np.array_equal(masks_to_xyxy(mask2), np.array([[2, 1, 4, 3]]))

    # Case with multiple masks
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1:4, 2:5] = True  # First rectangle
    masks[1, 0:3, 1:4] = True  # Second rectangle
    expected = np.array([[2, 1, 4, 3], [1, 0, 3, 2]])
    assert np.array_equal(masks_to_xyxy(masks), expected)

    # Case with a mask covering the entire image
    full_mask = np.ones((1, 5, 5), dtype=bool)
    assert np.array_equal(masks_to_xyxy(full_mask), np.array([[0, 0, 4, 4]]))


def test_det_post_process():
    cls_ids = np.array([[1, 2, 3]])
    boxes = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]])
    confs = np.array([[0.8, 0.9, 0.7]])
    out = [cls_ids, boxes, confs]

    im0_shape = (640, 480)
    conf_threshold = 0.75
    sv_detections = det_postprocess(out, im0_shape, conf_threshold)

    np.testing.assert_array_equal(sv_detections.xyxy, np.array([[48, 128, 144, 256], [240, 384, 336, 512]]))
    assert sv_detections.class_id is not None
    np.testing.assert_array_equal(sv_detections.class_id, np.array([1, 2]))
    assert sv_detections.confidence is not None
    np.testing.assert_array_equal(sv_detections.confidence, np.array([0.8, 0.9]))


def test_semseg_postprocess():
    cls_ids = np.array([1, 2, 3])
    mask = np.array(
        [
            [0, 1, 1, 2],
            [0, 1, 2, 2],
            [0, 0, 1, 2],
        ]
    )
    confs = np.array([0.7, 0.9, 0.8])
    out = [
        np.expand_dims(cls_ids, axis=0),
        np.expand_dims(mask, axis=0),
        np.expand_dims(confs, axis=0),
    ]

    im0_shape = (3, 4)
    conf_threshold = 0.75

    sv_detections = semseg_postprocess(out, im0_shape, conf_threshold)

    # Expected masks
    expected_masks = np.array(
        [
            [
                [False, True, True, False],
                [False, True, False, False],
                [False, False, True, False],
            ],  # Class 1
            [
                [False, False, False, True],
                [False, False, True, True],
                [False, False, False, True],
            ],  # Class 2
        ]
    )

    # Assertions
    assert sv_detections.mask is not None
    np.testing.assert_array_equal(sv_detections.mask, expected_masks)
    assert sv_detections.class_id is not None
    np.testing.assert_array_equal(sv_detections.class_id, np.array([2, 3]))
    assert sv_detections.confidence is not None
    np.testing.assert_array_equal(sv_detections.confidence, np.array([0.9, 0.8]))
    assert sv_detections.xyxy.shape == (2, 4)


def test_get_postprocess_fn():
    """
    Test the get_postprocess_fn function to ensure it returns
    the correct postprocessing function for each task.
    """
    # Test detection task
    det_fn = get_postprocess_fn(FocoosTask.DETECTION)
    assert det_fn == det_postprocess, "Detection task should return det_postprocess function"

    # Test instance segmentation task
    instance_fn = get_postprocess_fn(FocoosTask.INSTANCE_SEGMENTATION)
    assert instance_fn == instance_postprocess, "Instance segmentation task should return instance_postprocess function"

    # Test semantic segmentation task
    semseg_fn = get_postprocess_fn(FocoosTask.SEMSEG)
    assert semseg_fn == semseg_postprocess, "Semantic segmentation task should return semseg_postprocess function"

    # Test all FocoosTask values to ensure no exceptions
    for task in FocoosTask:
        fn = get_postprocess_fn(task)
        assert callable(fn), f"Postprocess function for {task} should be callable"


def test_instance_postprocess():
    """Test instance segmentation postprocessing"""
    cls_ids = np.array([0, 1])
    masks = np.zeros((2, 100, 100))
    masks[0, 10:30, 10:30] = 1
    masks[1, 40:60, 40:60] = 1
    confs = np.array([0.95, 0.85])
    out = [[cls_ids], [masks], [confs]]

    result = instance_postprocess(out, (100, 100), 0.8)

    assert isinstance(result, sv.Detections)
    assert len(result) == 2
    assert result.mask is not None
    assert result.xyxy is not None
    assert result.class_id is not None
    assert result.confidence is not None


def test_confidence_threshold_filtering():
    """Test that confidence threshold filtering works correctly"""
    out = [
        np.array([[0, 1, 2]]),  # cls_ids
        np.array([[[0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.6, 0.6], [0.7, 0.7, 0.9, 0.9]]]),  # boxes
        np.array([[0.95, 0.55, 0.85]]),  # confs
    ]

    result = det_postprocess(out, (100, 100), conf_threshold=0.8)

    assert len(result) == 2  # Should only keep detections with conf > 0.8
    assert all(conf > 0.8 for conf in result.confidence)
