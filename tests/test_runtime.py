import pathlib
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from focoos.ports import ModelMetadata, OnnxEngineOpts, RuntimeTypes
from focoos.runtime import ONNXRuntime, get_runtime


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
    mock_onnxruntime_class.return_value = MagicMock(
        spec=ONNXRuntime, opts=expected_opts
    )

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
