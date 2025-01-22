import pathlib
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from focoos.ports import ModelMetadata, OnnxEngineOpts, RuntimeTypes
from focoos.runtime import ONNXRuntime, det_postprocess, get_runtime, semseg_postprocess


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
            OnnxEngineOpts(
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
            OnnxEngineOpts(
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
            OnnxEngineOpts(
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
            OnnxEngineOpts(
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
            OnnxEngineOpts(
                cuda=False,
                trt=False,
                fp16=False,
                coreml=True,
                verbose=False,
                warmup_iter=2,
            ),
        ),
    ],
)
def test_get_run_time(mocker: MockerFixture, tmp_path, runtime_type, expected_opts):
    # mock model path
    model_path = pathlib.Path(tmp_path) / "fakeref" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()
    model_path = model_path.as_posix()

    # mock model metadata
    mock_model_metadata = MagicMock(spec=ModelMetadata)

    # mock opts
    mock_onnxruntime_class = mocker.patch("focoos.runtime.ONNXRuntime", autospec=True)
    mock_onnxruntime_class.return_value = MagicMock(spec=ONNXRuntime, opts=expected_opts)

    # warmup_iter
    warmup_iter = 2

    # call the function to test
    onnx_runtime = get_runtime(
        runtime_type=runtime_type,
        model_path=model_path,
        model_metadata=mock_model_metadata,
        warmup_iter=warmup_iter,
    )

    # assertions
    assert onnx_runtime is not None
    mock_onnxruntime_class.assert_called_once_with(
        model_path,
        expected_opts,
        mock_model_metadata,
    )
