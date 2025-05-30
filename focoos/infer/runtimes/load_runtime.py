from focoos.infer.runtimes.base import BaseRuntime
from focoos.ports import ModelInfo, OnnxRuntimeOpts, RuntimeType, TorchscriptRuntimeOpts
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
    runtime_type: RuntimeType,
    model_path: str,
    model_info: ModelInfo,
    warmup_iter: int = 50,
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
    if runtime_type == RuntimeType.TORCHSCRIPT_32:
        if not TORCH_AVAILABLE:
            logger.error(
                "⚠️ Pytorch not found =(  please install focoos with ['torch'] extra. See https://focoosai.github.io/focoos/setup/ for more details"
            )
            raise ImportError("Pytorch not found")
        from focoos.infer.runtimes.torchscript import TorchscriptRuntime

        opts = TorchscriptRuntimeOpts(warmup_iter=warmup_iter)
        return TorchscriptRuntime(model_path=model_path, opts=opts, model_info=model_info)
    else:
        if not ORT_AVAILABLE:
            logger.error(
                "⚠️ onnxruntime not found =(  please install focoos with one of 'onnx', 'onnx-cpu', extra. See https://focoosai.github.io/focoos/setup/ for more details"
            )
            raise ImportError("onnxruntime not found")
        from focoos.infer.runtimes.onnx import ONNXRuntime

        opts = OnnxRuntimeOpts(
            cuda=runtime_type == RuntimeType.ONNX_CUDA32,
            trt=runtime_type in [RuntimeType.ONNX_TRT32, RuntimeType.ONNX_TRT16],
            fp16=runtime_type == RuntimeType.ONNX_TRT16,
            warmup_iter=warmup_iter,
            coreml=runtime_type == RuntimeType.ONNX_COREML,
            verbose=False,
        )
    return ONNXRuntime(model_path=model_path, opts=opts, model_info=model_info)
