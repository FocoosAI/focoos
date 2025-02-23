import pathlib
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
from pytest_mock import MockerFixture

from focoos.ports import FocoosTask, ModelMetadata, OnnxRuntimeOpts, RuntimeTypes, TorchscriptRuntimeOpts
from focoos.runtime import (
    ORT_AVAILABLE,
    TORCH_AVAILABLE,
    ONNXRuntime,
    TorchscriptRuntime,
    det_postprocess,
    get_postprocess_fn,
    instance_postprocess,
    load_runtime,
    semseg_postprocess,
)


def test_runtime_availability():
    """
    Test the runtime availability flags.
    These flags should be boolean values indicating whether
    PyTorch and ONNX Runtime are available in the environment.
    """
    # Check that the flags are boolean
    assert isinstance(TORCH_AVAILABLE, bool), "TORCH_AVAILABLE should be a boolean"
    assert isinstance(ORT_AVAILABLE, bool), "ORT_AVAILABLE should be a boolean"

    # At least one runtime should be available for the library to work
    assert TORCH_AVAILABLE or ORT_AVAILABLE, "At least one runtime (PyTorch or ONNX Runtime) must be available"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_import():
    """
    Test PyTorch import when available.
    This test is skipped if PyTorch is not installed.
    """
    import torch

    assert torch is not None, "PyTorch should be properly imported"


@pytest.mark.skipif(not ORT_AVAILABLE, reason="ONNX Runtime not available")
def test_onnx_import():
    """
    Test ONNX Runtime import when available.
    This test is skipped if ONNX Runtime is not installed.
    """
    import onnxruntime as ort

    assert ort is not None, "ONNX Runtime should be properly imported"


def test_det_post_process():
    cls_ids = np.array([1, 2, 3])
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
    confs = np.array([0.8, 0.9, 0.7])
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


@pytest.mark.parametrize(
    "runtime_type, expected_opts",
    [
        (
            RuntimeTypes.ONNX_CUDA32,
            OnnxRuntimeOpts(
                cuda=True,
                trt=False,
                fp16=False,
                coreml=False,
                verbose=False,
                warmup_iter=2,
            ),
        ),
        (
            RuntimeTypes.ONNX_TRT32,
            OnnxRuntimeOpts(
                cuda=False,
                trt=True,
                fp16=False,
                coreml=False,
                verbose=False,
                warmup_iter=2,
            ),
        ),
        (
            RuntimeTypes.ONNX_TRT16,
            OnnxRuntimeOpts(
                cuda=False,
                trt=True,
                fp16=True,
                coreml=False,
                verbose=False,
                warmup_iter=2,
            ),
        ),
        (
            RuntimeTypes.ONNX_CPU,
            OnnxRuntimeOpts(
                cuda=False,
                trt=False,
                fp16=False,
                coreml=False,
                verbose=False,
                warmup_iter=2,
            ),
        ),
        (
            RuntimeTypes.ONNX_COREML,
            OnnxRuntimeOpts(
                cuda=False,
                trt=False,
                fp16=False,
                coreml=True,
                verbose=False,
                warmup_iter=2,
            ),
        ),
        (
            RuntimeTypes.TORCHSCRIPT_32,
            TorchscriptRuntimeOpts(
                warmup_iter=2,
                optimize_for_inference=True,
                set_fusion_strategy=True,
            ),
        ),
    ],
)
def test_load_runtime(mocker: MockerFixture, tmp_path, runtime_type, expected_opts):
    # mock model path
    model_path = pathlib.Path(tmp_path) / "fakeref" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()
    model_path = model_path.as_posix()

    # mock model metadata
    mock_model_metadata = MagicMock(spec=ModelMetadata)

    # mock opts
    if runtime_type == RuntimeTypes.TORCHSCRIPT_32:
        mocker.patch("focoos.runtime.TORCH_AVAILABLE", True)
        mock_runtime_class = mocker.patch("focoos.runtime.TorchscriptRuntime", autospec=True)
        mock_runtime_class.return_value = MagicMock(spec=TorchscriptRuntime, opts=expected_opts)
    else:
        mocker.patch("focoos.runtime.ORT_AVAILABLE", True)
        mock_runtime_class = mocker.patch("focoos.runtime.ONNXRuntime", autospec=True)
        mock_runtime_class.return_value = MagicMock(spec=ONNXRuntime, opts=expected_opts)

    # warmup_iter
    warmup_iter = 2

    # call the function to test
    runtime = load_runtime(
        runtime_type=runtime_type,
        model_path=model_path,
        model_metadata=mock_model_metadata,
        warmup_iter=warmup_iter,
    )

    # assertions
    assert runtime is not None
    mock_runtime_class.assert_called_once_with(
        model_path,
        expected_opts,
        mock_model_metadata,
    )


def test_load_unavailable_runtime(mocker: MockerFixture):
    mocker.patch("focoos.runtime.ORT_AVAILABLE", False)
    mocker.patch("focoos.runtime.TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        load_runtime(RuntimeTypes.TORCHSCRIPT_32, "fake_model_path", MagicMock(spec=ModelMetadata), 2)
    with pytest.raises(ImportError):
        load_runtime(RuntimeTypes.ONNX_CUDA32, "fake_model_path", MagicMock(spec=ModelMetadata), 2)


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
        np.array([0, 1, 2]),  # cls_ids
        np.array([[0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.6, 0.6], [0.7, 0.7, 0.9, 0.9]]),  # boxes
        np.array([0.95, 0.55, 0.85]),  # confs
    ]

    result = det_postprocess(out, (100, 100), conf_threshold=0.8)

    assert len(result) == 2  # Should only keep detections with conf > 0.8
    assert all(conf > 0.8 for conf in result.confidence)
