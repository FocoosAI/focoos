from pathlib import Path
from time import perf_counter
from typing import Union

import numpy as np
import onnxruntime as ort

# from supervision.detection.utils import mask_to_xyxy
from focoos.infer.runtimes.base import BaseRuntime
from focoos.ports import (
    LatencyMetrics,
    OnnxRuntimeOpts,
    RemoteModelInfo,
    Task,
)
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_gpu_info

GPU_ID = 0

logger = get_logger()


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
        model_info (RemoteModelInfo): Metadata about the model.
        ort_sess (ort.InferenceSession): ONNX Runtime inference session.
        active_providers (list): List of active execution providers.
        dtype (np.dtype): Input data type for the model.
    """

    def __init__(self, model_path: Union[str, Path], opts: OnnxRuntimeOpts, model_info: RemoteModelInfo):
        logger.debug(f"🔧 [onnxruntime device] {ort.get_device()}")

        self.name = Path(model_path).stem
        self.opts = opts
        self.model_info = model_info

        # Setup session options
        options = ort.SessionOptions()
        options.log_severity_level = 0 if opts.verbose else 2
        options.enable_profiling = opts.verbose

        # Setup providers
        self.providers = self._setup_providers(model_dir=Path(model_path).parent)
        self.active_provider = self.providers[0][0]
        logger.info(f"[onnxruntime] using: {self.active_provider}")
        # Create session
        self.ort_sess = ort.InferenceSession(model_path, options, providers=self.providers)

        if self.opts.trt and self.providers[0][0] == "TensorrtExecutionProvider":
            logger.info(
                "🟢 [onnxruntime] TensorRT enabled. First execution may take longer as it builds the TRT engine."
            )
        # Set input type
        self.dtype = np.uint8 if self.ort_sess.get_inputs()[0].type == "tensor(uint8)" else np.float32

        # Warmup
        if self.opts.warmup_iter > 0:
            self._warmup()

    def _setup_providers(self, model_dir: Path):
        providers = []
        available = ort.get_available_providers()
        logger.info(f"[onnxruntime] available providers:{available}")
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
                logger.warning(f"{provider} not found.")

        providers.append(("CPUExecutionProvider", {}))
        return providers

    def _warmup(self):
        size = self.model_info.im_size if self.model_info.task == Task.DETECTION and self.model_info.im_size else 640
        logger.info(f"⏱️ [onnxruntime] Warming up model {self.name} on {self.active_provider}, size: {size}x{size}..")
        np_image = np.random.rand(1, 3, size, size).astype(self.dtype)
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]

        for _ in range(self.opts.warmup_iter):
            self.ort_sess.run(out_name, {input_name: np_image})

        logger.info("⏱️ [onnxruntime] Warmup done")

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
            logger.warning(f"No GPU found, using CPU {device_name}.")

        logger.info(f"⏱️ [onnxruntime] Benchmarking latency on {device_name}, size: {size}x{size}..")
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
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics
