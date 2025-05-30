import json
from dataclasses import asdict
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from focoos.infer.infer_model import InferModel
from focoos.infer.runtimes.onnx import ONNXRuntime
from focoos.infer.runtimes.torchscript import TorchscriptRuntime
from focoos.ports import (
    FocoosDet,
    FocoosDetections,
    LatencyMetrics,
    ModelFamily,
    ModelInfo,
    RuntimeType,
    Task,
)


@pytest.fixture
def mock_model_info():
    """Fixture to provide a mock ModelInfo for testing."""
    return ModelInfo(
        name="test-model",
        model_family=ModelFamily.DETR,
        classes=["class_0", "class_1"],
        im_size=640,
        task=Task.DETECTION,
        config={},
        ref="test_model_ref",
        focoos_model="test_focoos_model",
        description="A test model for unit tests",
    )


@pytest.fixture
def mock_model_dir(tmp_path, mock_model_info: ModelInfo):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_info_path = model_dir / "model_info.json"
    model_info_path.write_text(json.dumps(asdict(mock_model_info)))
    (model_dir / "model.onnx").touch()
    (model_dir / "model.pt").touch()
    return model_dir


@pytest.fixture
def mock_local_model_onnx(mocker: MockerFixture, mock_model_dir):
    # Mock get_runtime
    mock_runtime = MagicMock(spec=ONNXRuntime)
    mock_get_runtime = mocker.patch("focoos.infer.infer_model.load_runtime")
    mock_get_runtime.return_value = mock_runtime

    # Mock processor and config manager
    mocker.patch("focoos.infer.infer_model.ProcessorManager.get_processor")
    mocker.patch("focoos.model_manager.ConfigManager.from_dict")

    model = InferModel(model_dir=mock_model_dir, runtime_type=RuntimeType.ONNX_CPU)
    return model


@pytest.fixture
def mock_local_model_torch(mocker: MockerFixture, mock_model_dir):
    # Mock get_runtime
    mock_runtime = MagicMock(spec=TorchscriptRuntime)
    mock_get_runtime = mocker.patch("focoos.infer.infer_model.load_runtime")
    mock_get_runtime.return_value = mock_runtime

    # Mock processor and config manager
    mocker.patch("focoos.infer.infer_model.ProcessorManager.get_processor")
    mocker.patch("focoos.model_manager.ConfigManager.from_dict")

    model = InferModel(model_dir=mock_model_dir, runtime_type=RuntimeType.TORCHSCRIPT_32)
    return model


def test_initialization_fail_no_model_dir():
    with pytest.raises(FileNotFoundError):
        InferModel(model_dir="fakedir", runtime_type=RuntimeType.ONNX_CPU)


def test_init_file_not_found(mocker: MockerFixture):
    mocker.patch("focoos.infer.infer_model.os.path.exists", return_value=False)
    with pytest.raises(FileNotFoundError):
        InferModel(model_dir="fakedir", runtime_type=RuntimeType.ONNX_CPU)


def test_initialization_onnx(mock_local_model_onnx: InferModel, mock_model_dir, mock_model_info):
    assert mock_local_model_onnx.model_dir == mock_model_dir
    assert mock_local_model_onnx.model_info.name == mock_model_info.name
    assert isinstance(mock_local_model_onnx.runtime, MagicMock)


def test_initialization_torch(mock_local_model_torch: InferModel, mock_model_dir, mock_model_info):
    assert mock_local_model_torch.model_dir == mock_model_dir
    assert mock_local_model_torch.model_info.name == mock_model_info.name
    assert isinstance(mock_local_model_torch.runtime, MagicMock)


def test_benchmark(mock_local_model_onnx: InferModel):
    mock_local_model_onnx.runtime.benchmark.return_value = MagicMock(spec=LatencyMetrics)
    iterations, size = 10, 1000

    result = mock_local_model_onnx.benchmark(iterations, size)

    assert result is not None
    assert isinstance(result, MagicMock)
    mock_local_model_onnx.runtime.benchmark.assert_called_once_with(iterations, (size, size))


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


def test_infer_onnx(
    mocker: MockerFixture,
    mock_local_model_onnx: InferModel,
    image_ndarray: np.ndarray,
    mock_focoos_detections: FocoosDetections,
):
    # Mock image preprocessing
    mock_image_preprocess = mocker.patch("focoos.infer.infer_model.image_preprocess")
    mock_image_preprocess.return_value = (image_ndarray, image_ndarray)

    # Mock processor methods
    mock_local_model_onnx.processor.preprocess = MagicMock(return_value=(MagicMock(), None))
    mock_local_model_onnx.processor.export_postprocess = MagicMock(return_value=[mock_focoos_detections])

    # Mock runtime call
    mock_local_model_onnx.runtime = MagicMock()
    mock_local_model_onnx.runtime.return_value = MagicMock()

    # Act
    result = mock_local_model_onnx.infer(image=image_ndarray, threshold=0.5)

    # Assertions
    assert result is not None
    assert isinstance(result, FocoosDetections)
    mock_image_preprocess.assert_called_once()
    mock_local_model_onnx.processor.preprocess.assert_called_once()
    mock_local_model_onnx.processor.export_postprocess.assert_called_once()


def test_infer_torch(
    mocker: MockerFixture,
    mock_local_model_torch: InferModel,
    image_ndarray: np.ndarray,
    mock_focoos_detections: FocoosDetections,
):
    # Mock image preprocessing
    mock_image_preprocess = mocker.patch("focoos.infer.infer_model.image_preprocess")
    mock_image_preprocess.return_value = (image_ndarray, image_ndarray)

    # Mock processor methods
    mock_local_model_torch.processor.preprocess = MagicMock(return_value=(MagicMock(), None))
    mock_local_model_torch.processor.export_postprocess = MagicMock(return_value=[mock_focoos_detections])

    # Mock runtime call
    mock_local_model_torch.runtime = MagicMock()
    mock_local_model_torch.runtime.return_value = MagicMock()

    # Act
    result = mock_local_model_torch.infer(image=image_ndarray, threshold=0.5)

    # Assertions
    assert result is not None
    assert isinstance(result, FocoosDetections)
    mock_image_preprocess.assert_called_once()
    mock_local_model_torch.processor.preprocess.assert_called_once()
    mock_local_model_torch.processor.export_postprocess.assert_called_once()


def test_call_method(
    mocker: MockerFixture,
    mock_local_model_onnx: InferModel,
    image_ndarray: np.ndarray,
    mock_focoos_detections: FocoosDetections,
):
    # Mock the infer method
    mock_infer = mocker.patch.object(mock_local_model_onnx, "infer", return_value=mock_focoos_detections)

    # Act
    result = mock_local_model_onnx(image=image_ndarray, threshold=0.5)

    # Assertions
    assert result is not None
    assert isinstance(result, FocoosDetections)
    mock_infer.assert_called_once_with(image_ndarray, 0.5)


def test_end2end_benchmark(mocker: MockerFixture, mock_local_model_onnx: InferModel):
    # Mock runtime.get_info
    mock_local_model_onnx.runtime.get_info = MagicMock(return_value=("ONNX", "CPU"))

    # Mock the infer method instead of __call__ to avoid the actual inference logic
    mock_infer = mocker.patch.object(mock_local_model_onnx, "infer", return_value=MagicMock())

    # Act
    result = mock_local_model_onnx.end2end_benchmark(iterations=5, size=640)

    # Assertions
    assert result is not None
    assert isinstance(result, LatencyMetrics)
    assert result.engine == "ONNX"
    assert result.device == "CPU"
    assert result.im_size == 640
    # The method adds 5 warmup iterations, so total calls = iterations + 5
    assert mock_infer.call_count == 10  # 5 iterations + 5 warmup iterations


def test_read_model_info_file_not_found(mocker: MockerFixture, tmp_path):
    # Create model directory without model_info.json
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Mock os.path.exists to return False for model.onnx check but True for directory
    def mock_exists(path):
        if "model.onnx" in str(path):
            return True
        if "model_info.json" in str(path):
            return False
        return True

    mocker.patch("focoos.infer.infer_model.os.path.exists", side_effect=mock_exists)

    with pytest.raises(FileNotFoundError, match="Model info file not found"):
        InferModel(model_dir=model_dir, runtime_type=RuntimeType.ONNX_CPU)


def test_benchmark_with_default_size(mock_local_model_onnx: InferModel):
    mock_local_model_onnx.runtime.benchmark.return_value = MagicMock(spec=LatencyMetrics)
    iterations = 10

    result = mock_local_model_onnx.benchmark(iterations)

    assert result is not None
    mock_local_model_onnx.runtime.benchmark.assert_called_once_with(
        iterations, (mock_local_model_onnx.model_info.im_size, mock_local_model_onnx.model_info.im_size)
    )


def test_benchmark_with_tuple_size(mock_local_model_onnx: InferModel):
    mock_local_model_onnx.runtime.benchmark.return_value = MagicMock(spec=LatencyMetrics)
    iterations, size = 10, (800, 600)

    result = mock_local_model_onnx.benchmark(iterations, size)

    assert result is not None
    mock_local_model_onnx.runtime.benchmark.assert_called_once_with(iterations, size)
