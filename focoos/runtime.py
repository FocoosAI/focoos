"""
Runtime Module for the models

This module provides the necessary functionality for loading, preprocessing,
running inference, and benchmarking ONNX and TorchScript models using different execution
providers such as CUDA, TensorRT, and CPU. It includes utility functions
for image preprocessing, postprocessing, and interfacing with the ONNXRuntime and TorchScript libraries.

Functions:
    det_postprocess: Postprocesses detection model outputs into sv.Detections.
    semseg_postprocess: Postprocesses semantic segmentation model outputs into sv.Detections.
    load_runtime: Returns an ONNXRuntime or TorchscriptRuntime instance configured for the given runtime type.

Classes:
    RuntimeTypes: Enum for the different runtime types.
    ONNXRuntime: A class that interfaces with ONNX Runtime for model inference.
    TorchscriptRuntime: A class that interfaces with TorchScript for model inference.
"""

from abc import abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# from supervision.detection.utils import mask_to_xyxy
from focoos.ports import (
    FocoosTask,
    LatencyMetrics,
    ModelMetadata,
    OnnxRuntimeOpts,
    RuntimeTypes,
    TorchscriptRuntimeOpts,
)
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_gpu_info

GPU_ID = 0

logger = get_logger()


class BaseRuntime:
    """
    Abstract base class for runtime implementations.

    This class defines the interface that all runtime implementations must follow.
    It provides methods for model initialization, inference, and performance benchmarking.

    Attributes:
        model_path (str): Path to the model file.
        opts (Any): Runtime-specific options.
        model_metadata (ModelMetadata): Metadata about the model.
    """

    def __init__(self, model_path: str, opts: Any, model_metadata: ModelMetadata):
        """
        Initialize the runtime with model path, options and metadata.

        Args:
            model_path (str): Path to the model file.
            opts (Any): Runtime-specific configuration options.
            model_metadata (ModelMetadata): Metadata about the model.
        """
        pass

    @abstractmethod
    def __call__(self, im: np.ndarray) -> np.ndarray:
        """
        Run inference on the input image.

        Args:
            im (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Model output as a numpy array.
        """
        pass

    @abstractmethod
    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        """
        Benchmark the model performance.

        Args:
            iterations (int, optional): Number of inference iterations to run. Defaults to 20.
            size (int, optional): Input image size for benchmarking. Defaults to 640.

        Returns:
            LatencyMetrics: Performance metrics including mean, median, and percentile latencies.
        """
        pass


class ONNXRuntime(BaseRuntime):
    """
    ONNX Runtime wrapper for model inference with different execution providers.

    This class implements the BaseRuntime interface for ONNX models, supporting
    various execution providers like CUDA, TensorRT, OpenVINO, and CoreML.
    It handles model initialization, provider configuration, warmup, inference,
    and performance benchmarking.

    Attributes:
        name (str): Name of the model derived from the model path.
        opts (OnnxRuntimeOpts): Configuration options for the ONNX runtime.
        model_metadata (ModelMetadata): Metadata about the model.
        ort_sess (ort.InferenceSession): ONNX Runtime inference session.
        active_providers (list): List of active execution providers.
        dtype (np.dtype): Input data type for the model.
    """

    def __init__(self, model_path: str, opts: OnnxRuntimeOpts, model_metadata: ModelMetadata):
        self.logger = get_logger()

        self.logger.debug(f"🔧 [onnxruntime device] {ort.get_device()}")

        self.name = Path(model_path).stem
        self.opts = opts
        self.model_metadata = model_metadata

        # Setup session options
        options = ort.SessionOptions()
        options.log_severity_level = 0 if opts.verbose else 2
        options.enable_profiling = opts.verbose

        # Setup providers
        self.providers = self._setup_providers(model_dir=Path(model_path).parent)
        self.active_provider = self.providers[0][0]
        self.logger.info(f"[onnxruntime] using: {self.active_provider}")
        # Create session
        self.ort_sess = ort.InferenceSession(model_path, options, providers=self.providers)

        if self.opts.trt and self.providers[0][0] == "TensorrtExecutionProvider":
            self.logger.info(
                "🟢 [onnxruntime] TensorRT enabled. First execution may take longer as it builds the TRT engine."
            )
        # Set input type
        self.dtype = np.uint8 if self.ort_sess.get_inputs()[0].type == "tensor(uint8)" else np.float32

        # Warmup
        if self.opts.warmup_iter > 0:
            self._warmup()

    def _setup_providers(self, model_dir: str):
        providers = []
        available = ort.get_available_providers()
        self.logger.info(f"[onnxruntime] available providers:{available}")
        _dir = Path(model_dir)
        models_root = _dir.parent
        # Check and add providers in order of preference
        provider_configs = [
            (
                "TensorrtExecutionProvider",
                self.opts.trt,
                {
                    "device_id": GPU_ID,
                    "trt_fp16_enable": self.opts.fp16,
                    "trt_force_sequential_engine_build": False,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(_dir / ".trt_cache"),
                    "trt_ep_context_file_path": str(_dir),
                    "trt_timing_cache_enable": True,  # Timing cache can be shared across multiple models if layers are the same
                    "trt_builder_optimization_level": 3,
                    "trt_timing_cache_path": str(models_root / ".trt_timing_cache"),
                },
            ),
            (
                "OpenVINOExecutionProvider",
                self.opts.vino,
                {"device_type": "MYRIAD_FP16", "enable_vpu_fast_compile": True, "num_of_threads": 1},
            ),
            (
                "CUDAExecutionProvider",
                self.opts.cuda,
                {
                    "device_id": GPU_ID,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 16 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            ("CoreMLExecutionProvider", self.opts.coreml, {}),
        ]

        for provider, enabled, config in provider_configs:
            if enabled and provider in available:
                providers.append((provider, config))
            elif enabled:
                self.logger.warning(f"{provider} not found.")

        providers.append(("CPUExecutionProvider", {}))
        return providers

    def _warmup(self):
        self.logger.info("⏱️ [onnxruntime] Warming up model ..")
        size = (
            self.model_metadata.im_size
            if self.model_metadata.task == FocoosTask.DETECTION and self.model_metadata.im_size
            else 640
        )
        np_image = np.random.rand(1, 3, size, size).astype(self.dtype)
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]

        for _ in range(self.opts.warmup_iter):
            self.ort_sess.run(out_name, {input_name: np_image})

        self.logger.info("⏱️ [onnxruntime] Warmup done")

    def __call__(self, im: np.ndarray) -> list[np.ndarray]:
        """
        Run inference on the input image.

        Args:
            im (np.ndarray): Input image as a numpy array.

        Returns:
            list[np.ndarray]: Model outputs as a list of numpy arrays.
        """
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]
        out = self.ort_sess.run(out_name, {input_name: im})
        return out

    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        """
        Benchmark the model performance.

        Runs multiple inference iterations and measures execution time to calculate
        performance metrics like FPS, mean latency, and other statistics.

        Args:
            iterations (int, optional): Number of inference iterations to run. Defaults to 20.
            size (int or tuple, optional): Input image size for benchmarking. Defaults to 640.

        Returns:
            LatencyMetrics: Performance metrics including FPS, mean, min, max, and std latencies.
        """
        gpu_info = get_gpu_info()
        device_name = "CPU"
        if gpu_info.devices is not None and len(gpu_info.devices) > 0:
            device_name = gpu_info.devices[0].gpu_name
        else:
            device_name = get_cpu_name()
            self.logger.warning(f"No GPU found, using CPU {device_name}.")

        self.logger.info(f"⏱️ [onnxruntime] Benchmarking latency on {device_name}..")
        size = size if isinstance(size, (tuple, list)) else (size, size)

        np_input = (255 * np.random.random((1, 3, size[0], size[1]))).astype(self.dtype)
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]

        durations = []
        for step in range(iterations + 5):
            start = perf_counter()
            self.ort_sess.run(out_name, {input_name: np_input})
            end = perf_counter()

            if step >= 5:  # Skip first 5 iterations
                durations.append((end - start) * 1000)

        durations = np.array(durations)

        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=f"onnx.{self.active_provider}",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size[0],
            device=str(device_name),
        )
        self.logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics


class TorchscriptRuntime(BaseRuntime):
    """
    TorchScript Runtime wrapper for model inference.

    This class implements the BaseRuntime interface for TorchScript models,
    supporting both CPU and CUDA devices. It handles model initialization,
    device placement, warmup, inference, and performance benchmarking.

    Attributes:
        device (torch.device): Device to run inference on (CPU or CUDA).
        opts (TorchscriptRuntimeOpts): Configuration options for the TorchScript runtime.
        model (torch.jit.ScriptModule): Loaded TorchScript model.
    """

    def __init__(
        self,
        model_path: str,
        opts: TorchscriptRuntimeOpts,
        model_metadata: ModelMetadata,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(name="TorchscriptEngine")
        self.logger.info(f"🔧 [torchscript] Device: {self.device}")
        self.opts = opts
        self.model_metadata = model_metadata
        map_location = None if torch.cuda.is_available() else "cpu"

        self.model = torch.jit.load(model_path, map_location=map_location)
        self.model = self.model.to(self.device)

        if self.opts.warmup_iter > 0:
            self.logger.info("⏱️ [torchscript] Warming up model..")
            with torch.no_grad():
                size = (
                    self.model_metadata.im_size
                    if self.model_metadata.task == FocoosTask.DETECTION and self.model_metadata.im_size
                    else 640
                )
                np_image = torch.rand(1, 3, size, size, device=self.device)
                for _ in range(self.opts.warmup_iter):
                    self.model(np_image)
            self.logger.info("⏱️ [torchscript] WARMUP DONE")

    def __call__(self, im: np.ndarray) -> list[np.ndarray]:
        """
        Run inference on the input image.

        Args:
            im (np.ndarray): Input image as a numpy array.

        Returns:
            list[np.ndarray]: Model outputs as a list of numpy arrays.
        """
        with torch.no_grad():
            torch_image = torch.from_numpy(im).to(self.device, dtype=torch.float32)
            res = self.model(torch_image)
            return [r.cpu().numpy() for r in res]

    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        """
        Benchmark the model performance.

        Runs multiple inference iterations and measures execution time to calculate
        performance metrics like FPS, mean latency, and other statistics.

        Args:
            iterations (int, optional): Number of inference iterations to run. Defaults to 20.
            size (int or tuple, optional): Input image size for benchmarking. Defaults to 640.

        Returns:
            LatencyMetrics: Performance metrics including FPS, mean, min, max, and std latencies.
        """
        gpu_info = get_gpu_info()
        device_name = "CPU"
        if gpu_info.devices is not None and len(gpu_info.devices) > 0:
            device_name = gpu_info.devices[0].gpu_name
        else:
            device_name = get_cpu_name()
            self.logger.warning(f"No GPU found, using CPU {device_name}.")
        self.logger.info("⏱️ [torchscript] Benchmarking latency..")
        size = size if isinstance(size, (tuple, list)) else (size, size)

        torch_input = torch.rand(1, 3, size[0], size[1], device=self.device)
        durations = []

        with torch.no_grad():
            for step in range(iterations + 5):
                start = perf_counter()
                self.model(torch_input)
                end = perf_counter()

                if step >= 5:  # Skip first 5 iterations
                    durations.append((end - start) * 1000)

        durations = np.array(durations)

        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean().astype(float)),
            engine="torchscript",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size[0],
            device=str(device_name),
        )
        self.logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics


def load_runtime(
    runtime_type: RuntimeTypes,
    model_path: str,
    model_metadata: ModelMetadata,
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
        opts = TorchscriptRuntimeOpts(warmup_iter=warmup_iter)
        return TorchscriptRuntime(model_path, opts, model_metadata)
    else:
        if not ORT_AVAILABLE:
            logger.error(
                "⚠️ onnxruntime not found =(  please install focoos with one of 'cpu', 'cuda', 'tensorrt' extra. See https://focoosai.github.io/focoos/setup/ for more details"
            )
            raise ImportError("onnxruntime not found")
        opts = OnnxRuntimeOpts(
            cuda=runtime_type == RuntimeTypes.ONNX_CUDA32,
            trt=runtime_type in [RuntimeTypes.ONNX_TRT32, RuntimeTypes.ONNX_TRT16],
            fp16=runtime_type == RuntimeTypes.ONNX_TRT16,
            warmup_iter=warmup_iter,
            coreml=runtime_type == RuntimeTypes.ONNX_COREML,
            verbose=False,
        )
    return ONNXRuntime(model_path, opts, model_metadata)
