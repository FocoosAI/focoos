import pathlib
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from focoos.ports import ModelInfo, OnnxRuntimeOpts, RuntimeTypes, TorchscriptRuntimeOpts
from focoos.runtime import (
    ORT_AVAILABLE,
    TORCH_AVAILABLE,
    ONNXRuntime,
    TorchscriptRuntime,
    load_runtime,
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
    mock_model_metadata = MagicMock(spec=ModelInfo)

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
        load_runtime(RuntimeTypes.TORCHSCRIPT_32, "fake_model_path", MagicMock(spec=ModelInfo), 2)
    with pytest.raises(ImportError):
        load_runtime(RuntimeTypes.ONNX_CUDA32, "fake_model_path", MagicMock(spec=ModelInfo), 2)
