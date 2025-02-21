from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
from pytest_mock import MockerFixture

from focoos.local_model import LocalModel
from focoos.ports import (
    FocoosDet,
    FocoosDetections,
    FocoosTask,
    LatencyMetrics,
    ModelMetadata,
    RuntimeTypes,
)
from focoos.runtime import ONNXRuntime, TorchscriptRuntime


@pytest.fixture
def mock_model_dir(tmp_path, mock_metadata: ModelMetadata):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metadata_path = model_dir / "focoos_metadata.json"
    metadata_path.write_text(mock_metadata.model_dump_json())
    (model_dir / "model.onnx").touch()
    return model_dir


@pytest.fixture
def mock_local_model_onnx(mocker: MockerFixture, mock_model_dir, image_ndarray):
    # Mock get_runtime
    mock_runtime = MagicMock(spec=ONNXRuntime)
    mock_get_runtime = mocker.patch("focoos.local_model.load_runtime", mock_runtime)
    mock_get_runtime.return_value = mock_runtime
    mocker.patch("focoos.local_model.os.path.exists", return_value=True)
    model = LocalModel(model_dir=mock_model_dir, runtime_type=RuntimeTypes.ONNX_CPU)

    # Mock BoxAnnotator
    mock_box_annotator = mocker.patch("focoos.local_model.sv.BoxAnnotator", autospec=True)
    mock_box_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock LabelAnnotator
    mock_label_annotator = mocker.patch("focoos.local_model.sv.LabelAnnotator", autospec=True)
    mock_label_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock MaskAnnotator
    mock_mask_annotator = mocker.patch("focoos.local_model.sv.MaskAnnotator", autospec=True)
    mock_mask_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Inject mock annotators into the local model
    model.box_annotator = mock_box_annotator
    model.label_annotator = mock_label_annotator
    model.mask_annotator = mock_mask_annotator
    return model


@pytest.fixture
def mock_local_model_torch(mocker: MockerFixture, mock_model_dir, image_ndarray):
    # Mock get_runtime
    mock_runtime = MagicMock(spec=TorchscriptRuntime)
    mock_get_runtime = mocker.patch("focoos.local_model.load_runtime", mock_runtime)
    mock_get_runtime.return_value = mock_runtime
    mocker.patch("focoos.local_model.os.path.exists", return_value=True)
    model = LocalModel(model_dir=mock_model_dir, runtime_type=RuntimeTypes.TORCHSCRIPT_32)

    # Mock BoxAnnotator
    mock_box_annotator = mocker.patch("focoos.local_model.sv.BoxAnnotator", autospec=True)
    mock_box_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock LabelAnnotator
    mock_label_annotator = mocker.patch("focoos.local_model.sv.LabelAnnotator", autospec=True)
    mock_label_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock MaskAnnotator
    mock_mask_annotator = mocker.patch("focoos.local_model.sv.MaskAnnotator", autospec=True)
    mock_mask_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Inject mock annotators into the local model
    model.box_annotator = mock_box_annotator
    model.label_annotator = mock_label_annotator
    model.mask_annotator = mock_mask_annotator
    return model


def test_initialization_fail_no_model_dir():
    with pytest.raises(FileNotFoundError):
        LocalModel(model_dir="fakedir", runtime_type=RuntimeTypes.ONNX_CPU)


def test_init_file_not_found(mocker: MockerFixture):
    mocker.patch("focoos.local_model.os.path.exists", return_value=False)
    with pytest.raises(FileNotFoundError):
        LocalModel(model_dir="fakedir", runtime_type=RuntimeTypes.ONNX_CPU)


def test_initialization_onnx(mock_local_model_onnx: LocalModel, mock_model_dir, mock_metadata):
    assert mock_local_model_onnx.model_dir == mock_model_dir
    assert mock_local_model_onnx.metadata == mock_metadata
    assert isinstance(mock_local_model_onnx.runtime, ONNXRuntime)


def test_initialization_torch(mock_local_model_torch: LocalModel, mock_model_dir, mock_metadata):
    assert mock_local_model_torch.model_dir == mock_model_dir
    assert mock_local_model_torch.metadata == mock_metadata
    assert isinstance(mock_local_model_torch.runtime, TorchscriptRuntime)


def test_benchmark(mock_local_model_onnx: LocalModel):
    mock_local_model_onnx.runtime.benchmark.return_value = MagicMock(spec=LatencyMetrics)
    iterations, size = 10, 1000

    result = mock_local_model_onnx.benchmark(iterations, size)

    assert result is not None
    assert isinstance(result, LatencyMetrics)
    mock_local_model_onnx.runtime.benchmark.assert_called_once_with(iterations, size)


@pytest.fixture
def mock_focoos_detections():
    mock_detection = FocoosDet(
        cls_id=0,
        conf=0.95,
        bbox=[10, 10, 50, 50],
    )
    return FocoosDetections(
        detections=[mock_detection],
        latency={"inference": 0.1, "preprocess": 0.05, "postprocess": 0.02},
    )


@pytest.fixture
def mock_sv_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[2, 8, 16, 32], [4, 10, 18, 34]]),
        class_id=np.array([0, 1]),
        confidence=np.array([0.8, 0.9]),
    )


def test_annotate_detection_metadata_classes_none(
    image_ndarray: np.ndarray, mock_local_model_onnx: LocalModel, mock_sv_detections
):
    mock_local_model_onnx.metadata.classes = None
    annotated_im = mock_local_model_onnx._annotate(image_ndarray, mock_sv_detections)
    assert annotated_im is not None
    assert isinstance(annotated_im, np.ndarray)
    mock_local_model_onnx.box_annotator.annotate.assert_called_once()
    mock_local_model_onnx.label_annotator.annotate.assert_called_once()
    mock_local_model_onnx.mask_annotator.annotate.assert_not_called()


def test_annotate_detection(image_ndarray: np.ndarray, mock_local_model_onnx: LocalModel, mock_sv_detections):
    annotated_im = mock_local_model_onnx._annotate(image_ndarray, mock_sv_detections)
    assert annotated_im is not None
    assert isinstance(annotated_im, np.ndarray)
    mock_local_model_onnx.box_annotator.annotate.assert_called_once()
    mock_local_model_onnx.label_annotator.annotate.assert_called_once()
    mock_local_model_onnx.mask_annotator.annotate.assert_not_called()


def test_annotate_semseg(image_ndarray: np.ndarray, mock_local_model_onnx: LocalModel, mock_sv_detections):
    mock_local_model_onnx.metadata.task = FocoosTask.SEMSEG
    annotated_im = mock_local_model_onnx._annotate(image_ndarray, mock_sv_detections)
    assert annotated_im is not None
    assert isinstance(annotated_im, np.ndarray)
    mock_local_model_onnx.box_annotator.annotate.asser_not_called()
    mock_local_model_onnx.label_annotator.annotate.asser_not_called()
    mock_local_model_onnx.mask_annotator.annotate.assert_called_once()


def mock_infer_setup(
    mocker: MockerFixture,
    mock_local_model: LocalModel,
    image_ndarray: np.ndarray,
    mock_sv_detections: sv.Detections,
    mock_focoos_detections: FocoosDetections,
    annotate: bool,
):
    """Setup for mocking the infer method."""
    # Mock image_preprocess
    mock_image_preprocess = mocker.patch("focoos.local_model.image_preprocess")
    mock_image_preprocess.return_value = (image_ndarray, image_ndarray)

    # Mock scale_detections
    mock_scale_detections = mocker.patch("focoos.local_model.scale_detections")
    mock_scale_detections.return_value = mock_sv_detections

    # Mock sv_to_focoos_detections
    mock_sv_to_focoos_detections = mocker.patch("focoos.local_model.sv_to_fai_detections")
    mock_sv_to_focoos_detections.return_value = mock_focoos_detections.detections

    # Mock _annotate
    mock_annotate = mocker.patch.object(mock_local_model, "_annotate", autospec=True)
    if annotate:
        mock_annotate.return_value = image_ndarray
    else:
        mock_annotate.return_value = None  # No annotation if False

    # Mock runtime
    class MockRuntime(MagicMock):
        def __call__(self, *args, **kwargs):
            return mock_sv_detections

    mock_runtime_call = mocker.patch.object(MockRuntime, "__call__", return_value=mock_sv_detections)
    mock_local_model.runtime = MockRuntime(spec=ONNXRuntime)

    return (
        mock_image_preprocess,
        mock_runtime_call,
        mock_scale_detections,
        mock_sv_to_focoos_detections,
        mock_annotate,
    )


@pytest.mark.parametrize("annotate", [(False, None)])
def test_infer_(
    mocker,
    mock_local_model_onnx,
    image_ndarray,
    mock_sv_detections,
    mock_focoos_detections,
    annotate,
):
    # Arrange
    *mock_to_call_once, mock_annotate = mock_infer_setup(
        mocker,
        mock_local_model_onnx,
        image_ndarray,
        mock_sv_detections,
        mock_focoos_detections,
        annotate,
    )

    # Act
    out, im = mock_local_model_onnx.infer(image=image_ndarray, annotate=annotate)

    # Assertions
    assert out is not None
    assert isinstance(out, FocoosDetections)

    for mock_obj in mock_to_call_once:
        mock_obj.assert_called_once()
    if annotate:
        mock_annotate.assert_called_once()
        assert im is not None
        assert isinstance(im, np.ndarray)
    else:
        mock_annotate.assert_not_called()
        assert im is None
