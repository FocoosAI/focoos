from focoos.infer.runtimes.base import BaseRuntime
from focoos.ports import OnnxRuntimeOpts, RemoteModelInfo, RuntimeTypes, TorchscriptRuntimeOpts
from focoos.utils.logger import get_logger

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort  # noqa: F401

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


logger = get_logger()


def load_runtime(
    runtime_type: RuntimeTypes,
    model_path: str,
    model_metadata: RemoteModelInfo,
    warmup_iter: int = 0,
) -> BaseRuntime:
    """
    Creates and returns a runtime instance based on the specified runtime type.
    Supports both ONNX and TorchScript runtimes with various execution providers.

    Args:
        runtime_type (RuntimeTypes): The type of runtime to use. Can be one of:
            - ONNX_CUDA32: ONNX runtime with CUDA FP32
            - ONNX_TRT32: ONNX runtime with TensorRT FP32
            - ONNX_TRT16: ONNX runtime with TensorRT FP16
            - ONNX_CPU: ONNX runtime with CPU
            - ONNX_COREML: ONNX runtime with CoreML
            - TORCHSCRIPT_32: TorchScript runtime with FP32
        model_path (str): Path to the model file (.onnx or .pt)
        model_metadata (ModelMetadata): Model metadata containing task type, classes etc.
        warmup_iter (int, optional): Number of warmup iterations before inference. Defaults to 0.

    Returns:
        BaseRuntime: A configured runtime instance (ONNXRuntime or TorchscriptRuntime)

    Raises:
        ImportError: If required dependencies (torch/onnxruntime) are not installed
    """
    if runtime_type == RuntimeTypes.TORCHSCRIPT_32:
        if not TORCH_AVAILABLE:
            logger.error(
                "⚠️ Pytorch not found =(  please install focoos with ['torch'] extra. See https://focoosai.github.io/focoos/setup/ for more details"
            )
            raise ImportError("Pytorch not found")
        from focoos.infer.runtimes.torchscript import TorchscriptRuntime

        opts = TorchscriptRuntimeOpts(warmup_iter=warmup_iter)
        return TorchscriptRuntime(model_path, opts, model_metadata)
    else:
        if not ORT_AVAILABLE:
            logger.error(
                "⚠️ onnxruntime not found =(  please install focoos with one of 'cpu', 'cuda', 'tensorrt' extra. See https://focoosai.github.io/focoos/setup/ for more details"
            )
            raise ImportError("onnxruntime not found")
        from focoos.infer.runtimes.onnx import ONNXRuntime

        opts = OnnxRuntimeOpts(
            cuda=runtime_type == RuntimeTypes.ONNX_CUDA32,
            trt=runtime_type in [RuntimeTypes.ONNX_TRT32, RuntimeTypes.ONNX_TRT16],
            fp16=runtime_type == RuntimeTypes.ONNX_TRT16,
            warmup_iter=warmup_iter,
            coreml=runtime_type == RuntimeTypes.ONNX_COREML,
            verbose=False,
        )
    return ONNXRuntime(model_path, opts, model_metadata)
