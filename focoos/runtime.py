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
from typing import Any, List, Tuple

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

import supervision as sv

from focoos.ports import (
    FocoosTask,
    LatencyMetrics,
    ModelMetadata,
    OnnxRuntimeOpts,
    RuntimeTypes,
    TorchscriptRuntimeOpts,
)
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_gpu_name

GPU_ID = 0

logger = get_logger()


def get_postprocess_fn(task: FocoosTask):
    if task == FocoosTask.INSTANCE_SEGMENTATION:
        return instance_postprocess
    elif task == FocoosTask.SEMSEG:
        return semseg_postprocess
    else:
        return det_postprocess


def det_postprocess(out: List[np.ndarray], im0_shape: Tuple[int, int], conf_threshold: float) -> sv.Detections:
    """
    Postprocesses the output of an object detection model and filters detections
    based on a confidence threshold.

    Args:
        out (List[np.ndarray]): The output of the detection model.
        im0_shape (Tuple[int, int]): The original shape of the input image (height, width).
        conf_threshold (float): The confidence threshold for filtering detections.

    Returns:
        sv.Detections: A sv.Detections object containing the filtered bounding boxes, class ids, and confidences.
    """
    cls_ids, boxes, confs = out
    boxes[:, 0::2] *= im0_shape[1]
    boxes[:, 1::2] *= im0_shape[0]
    high_conf_indices = (confs > conf_threshold).nonzero()

    return sv.Detections(
        xyxy=boxes[high_conf_indices].astype(int),
        class_id=cls_ids[high_conf_indices].astype(int),
        confidence=confs[high_conf_indices].astype(float),
    )


def semseg_postprocess(out: List[np.ndarray], im0_shape: Tuple[int, int], conf_threshold: float) -> sv.Detections:
    """
    Postprocesses the output of a semantic segmentation model and filters based
    on a confidence threshold.

    Args:
        out (List[np.ndarray]): The output of the semantic segmentation model.
        conf_threshold (float): The confidence threshold for filtering detections.

    Returns:
        sv.Detections: A sv.Detections object containing the masks, class ids, and confidences.
    """
    cls_ids, mask, confs = out[0][0], out[1][0], out[2][0]
    masks = np.equal(mask, np.arange(len(cls_ids))[:, None, None])
    high_conf_indices = np.where(confs > conf_threshold)[0]
    masks = masks[high_conf_indices].astype(bool)
    cls_ids = cls_ids[high_conf_indices].astype(int)
    confs = confs[high_conf_indices].astype(float)
    return sv.Detections(
        mask=masks,
        # xyxy is required from supervision
        xyxy=np.zeros(shape=(len(high_conf_indices), 4), dtype=np.uint8),
        class_id=cls_ids,
        confidence=confs,
    )


def instance_postprocess(out: List[np.ndarray], im0_shape: Tuple[int, int], conf_threshold: float) -> sv.Detections:
    """
    Postprocesses the output of an instance segmentation model and filters detections
    based on a confidence threshold.
    """
    cls_ids, mask, confs = out[0][0], out[1][0], out[2][0]
    high_conf_indices = np.where(confs > conf_threshold)[0]

    masks = mask[high_conf_indices].astype(bool)
    cls_ids = cls_ids[high_conf_indices].astype(int)
    confs = confs[high_conf_indices].astype(float)
    return sv.Detections(
        mask=masks,
        # xyxy is required from supervision
        xyxy=np.zeros(shape=(len(high_conf_indices), 4), dtype=np.uint8),
        class_id=cls_ids,
        confidence=confs,
    )


class BaseRuntime:
    def __init__(self, model_path: str, opts: Any, model_metadata: ModelMetadata):
        pass

    @abstractmethod
    def __call__(self, im: np.ndarray, conf_threshold: float) -> sv.Detections:
        pass

    @abstractmethod
    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        pass


class ONNXRuntime(BaseRuntime):
    """
    ONNX Runtime wrapper for model inference with different execution providers.
    Handles preprocessing, inference, postprocessing and benchmarking.
    """

    def __init__(self, model_path: str, opts: OnnxRuntimeOpts, model_metadata: ModelMetadata):
        self.logger = get_logger()

        self.logger.debug(f"🔧 [onnxruntime device] {ort.get_device()}")
        self.logger.debug(f"🔧 [onnxruntime available providers] {ort.get_available_providers()}")

        self.name = Path(model_path).stem
        self.opts = opts
        self.model_metadata = model_metadata

        self.postprocess_fn = get_postprocess_fn(model_metadata.task)

        # Setup session options
        options = ort.SessionOptions()
        options.log_severity_level = 0 if opts.verbose else 2
        options.enable_profiling = opts.verbose

        # Setup providers
        providers = self._setup_providers()

        # Create session
        self.ort_sess = ort.InferenceSession(model_path, options, providers=providers)
        self.active_providers = self.ort_sess.get_providers()
        self.logger.info(f"[onnxruntime] Active providers:{self.active_providers}")

        # Set input type
        self.dtype = np.uint8 if self.ort_sess.get_inputs()[0].type == "tensor(uint8)" else np.float32

        # Warmup
        if self.opts.warmup_iter > 0:
            self._warmup()

    def _setup_providers(self):
        providers = []
        available = ort.get_available_providers()

        # Check and add providers in order of preference
        provider_configs = [
            (
                "TensorrtExecutionProvider",
                self.opts.trt,
                {"device_id": 0, "trt_fp16_enable": self.opts.fp16, "trt_force_sequential_engine_build": False},
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

        providers.append("CPUExecutionProvider")
        return providers

    def _warmup(self):
        self.logger.info("⏱️ [onnxruntime] Warming up model ..")
        np_image = np.random.rand(1, 3, 640, 640).astype(self.dtype)
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]

        for _ in range(self.opts.warmup_iter):
            self.ort_sess.run(out_name, {input_name: np_image})

        self.logger.info("⏱️ [onnxruntime] Warmup done")

    def __call__(self, im: np.ndarray, conf_threshold: float) -> sv.Detections:
        """Run inference and return detections."""
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]
        out = self.ort_sess.run(out_name, {input_name: im})
        return self.postprocess_fn(out=out, im0_shape=(im.shape[2], im.shape[3]), conf_threshold=conf_threshold)

    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        """Benchmark model latency."""
        self.logger.info("⏱️ [onnxruntime] Benchmarking latency..")
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
        provider = self.active_providers[0]
        device = (
            get_gpu_name() if provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"] else get_cpu_name()
        )

        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=f"onnx.{provider}",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size[0],
            device=str(device),
        )
        self.logger.info(f"🔥 FPS: {metrics.fps}")
        return metrics


class TorchscriptRuntime(BaseRuntime):
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
        self.postprocess_fn = get_postprocess_fn(model_metadata.task)

        map_location = None if torch.cuda.is_available() else "cpu"

        self.model = torch.jit.load(model_path, map_location=map_location)
        self.model = self.model.to(self.device)

        if self.opts.warmup_iter > 0:
            self.logger.info("⏱️ [torchscript] Warming up model..")
            with torch.no_grad():
                np_image = torch.rand(1, 3, 640, 640, device=self.device)
                for _ in range(self.opts.warmup_iter):
                    self.model(np_image)
            self.logger.info("⏱️ [torchscript] WARMUP DONE")

    def __call__(self, im: np.ndarray, conf_threshold: float) -> sv.Detections:
        """Run inference and return detections."""
        with torch.no_grad():
            torch_image = torch.from_numpy(im).to(self.device, dtype=torch.float32)
            res = self.model(torch_image)
            return self.postprocess_fn([r.cpu().numpy() for r in res], (im.shape[2], im.shape[3]), conf_threshold)

    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        """Benchmark model latency."""
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
        device = get_gpu_name() if torch.cuda.is_available() else get_cpu_name()

        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean().astype(float)),
            engine="torchscript",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size[0],
            device=str(device),
        )
        self.logger.info(f"🔥 FPS: {metrics.fps}")
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
