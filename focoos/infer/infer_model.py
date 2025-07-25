"""
InferModel Module

This module provides the `InferModel` class that allows loading, inference,
and benchmark testing of models in a local environment. It supports detection
and segmentation tasks, and utilizes various runtime backends including ONNXRuntime
and TorchScript for model execution.

Classes:
    InferModel: A class for managing and interacting with local models.

Methods:
    __init__: Initializes the InferModel instance, loading the model, metadata,
              and setting up the runtime.
    _read_metadata: Reads the model metadata from a JSON file.
    _annotate: Annotates the input image with detection or segmentation results.
    infer: Runs inference on an input image, with optional annotation.
    benchmark: Benchmarks the model's inference performance over a specified
               number of iterations and input size.
"""

import os
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple, Union

import numpy as np
import supervision as sv
from PIL import Image

from focoos.config import FOCOOS_CONFIG
from focoos.infer.runtimes.base import BaseRuntime
from focoos.infer.runtimes.load_runtime import load_runtime
from focoos.ports import (
    FocoosDetections,
    LatencyMetrics,
    ModelExtension,
    ModelInfo,
    RuntimeType,
)
from focoos.processor.processor_manager import ProcessorManager
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_device_name
from focoos.utils.vision import (
    image_preprocess,
)

logger = get_logger("InferModel")


class InferModel:
    def __init__(
        self,
        model_dir: Union[str, Path],
        runtime_type: Optional[RuntimeType] = None,
    ):
        """
        Initialize a LocalModel instance.

        This class sets up a local model for inference by initializing the runtime environment,
        loading metadata, and preparing annotation utilities.

        Args:
            model_dir (Union[str, Path]): The path to the directory containing the model files.
            runtime_type (Optional[RuntimeTypes]): Specifies the runtime type to use for inference.
                Defaults to the value of `FOCOOS_CONFIG.runtime_type` if not provided.

        Raises:
            ValueError: If no runtime type is provided and `FOCOOS_CONFIG.runtime_type` is not set.
            FileNotFoundError: If the specified model directory does not exist.

        Attributes:
            model_dir (Union[str, Path]): Path to the model directory.
            metadata (ModelMetadata): Metadata information for the model.
            model_ref: Reference identifier for the model obtained from metadata.
            label_annotator (sv.LabelAnnotator): Utility for adding labels to the output,
                initialized with text padding and border radius.
            box_annotator (sv.BoxAnnotator): Utility for annotating bounding boxes.
            mask_annotator (sv.MaskAnnotator): Utility for annotating masks.
            runtime (ONNXRuntime): Inference runtime initialized with the specified runtime type,
                model path, metadata, and warmup iterations.

        The method verifies the existence of the model directory, reads the model metadata,
        and initializes the runtime for inference using the provided runtime type. Annotation
        utilities are also prepared for visualizing model outputs.
        """

        # Determine runtime type and model format
        runtime_type = runtime_type or FOCOOS_CONFIG.runtime_type
        extension = ModelExtension.from_runtime_type(runtime_type)

        # Set model directory and path
        self.model_dir: Union[str, Path] = model_dir
        self.model_path = os.path.join(model_dir, f"model.{extension.value}")
        logger.debug(f"Runtime type: {runtime_type}, Loading model from {self.model_path}..")

        # Check if model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        # Load metadata and set model reference
        # self.metadata: RemoteModelInfo = self._read_metadata()

        self.model_info: ModelInfo = self._read_model_info()

        try:
            from focoos.model_manager import ConfigManager

            model_config = ConfigManager.from_dict(self.model_info.model_family, self.model_info.config)
            self.processor = ProcessorManager.get_processor(self.model_info.model_family, model_config)
        except Exception as e:
            logger.error(f"Error creating model config: {e}")
            raise e

        # Initialize annotation utilities
        self.label_annotator = sv.LabelAnnotator(text_padding=10, border_radius=10)
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

        # Load runtime for inference
        self.runtime: BaseRuntime = load_runtime(
            runtime_type,
            str(self.model_path),
            self.model_info,
            FOCOOS_CONFIG.warmup_iter,
        )

    def _read_model_info(self) -> ModelInfo:
        """
        Reads the model info from a JSON file.
        """
        model_info_path = os.path.join(self.model_dir, "model_info.json")
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"Model info file not found: {model_info_path}")
        return ModelInfo.from_json(model_info_path)

    def __call__(
        self, image: Union[bytes, str, Path, np.ndarray, Image.Image], threshold: float = 0.5
    ) -> FocoosDetections:
        return self.infer(image, threshold)

    def infer(
        self,
        image: Union[bytes, str, Path, np.ndarray, Image.Image],
        threshold: float = 0.5,
    ) -> FocoosDetections:
        """
        Run inference on an input image and optionally annotate the results.

        Args:
            image (Union[bytes, str, Path, np.ndarray, Image.Image]): The input image to infer on.
                This can be a byte array, file path, or a PIL Image object, or a NumPy array representing the image.
            threshold (float, optional): The confidence threshold for detections. Defaults to 0.5.
                Detections with confidence scores below this threshold will be discarded.

        Returns:
            FocoosDetections: The detections from the inference, represented as a custom object (`FocoosDetections`).

        Raises:
            ValueError: If the model is not deployed locally (i.e., `self.runtime` is `None`).

        Example:
            ```python
            from focoos import Focoos, LocalModel

            focoos = Focoos()
            model = focoos.get_local_model(model_ref="<model_ref>")
            detections = model.infer(image, threshold=0.5)
            ```
        """
        assert self.runtime is not None, "Model is not deployed (locally)"

        t0 = perf_counter()
        im1, im0 = image_preprocess(image)
        tensors, _ = self.processor.preprocess(inputs=im1, device="cuda", image_size=self.model_info.im_size)
        logger.debug(f"Input image size: {im0.shape}")
        t1 = perf_counter()

        raw_detections = self.runtime(tensors)

        t2 = perf_counter()
        detections = self.processor.export_postprocess(
            raw_detections, im0, threshold=threshold, class_names=self.model_info.classes
        )
        t3 = perf_counter()
        latency = {
            "inference": round(t2 - t1, 3),
            "preprocess": round(t1 - t0, 3),
            "postprocess": round(t3 - t2, 3),
        }
        res = detections[0]  #!TODO  check for batching
        res.latency = latency

        logger.debug(
            f"Found {len(res)} detections. thr: {threshold} Inference time: {(t2 - t1) * 1000:.0f}ms, preprocess: {(t1 - t0) * 1000:.0f}ms, postprocess: {(t3 - t2) * 1000:.0f}ms"
        )
        return res

    def benchmark(self, iterations: int = 50, size: Optional[Union[int, Tuple[int, int]]] = None) -> LatencyMetrics:
        """
        Benchmark the model's inference performance over multiple iterations.
        """
        if size is None:
            size = self.model_info.im_size
        if isinstance(size, int):
            size = (size, size)
        return self.runtime.benchmark(iterations, size)

    def end2end_benchmark(
        self, iterations: int = 50, size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> LatencyMetrics:
        """
        Benchmark the model's inference performance over multiple iterations.

        Args:
            iterations (int): Number of iterations to run for benchmarking.
            size (int): The input size for each benchmark iteration.

        Returns:
            LatencyMetrics: Latency metrics including time taken for inference.

        Example:
            ```python
            from focoos import Focoos, LocalModel

            focoos = Focoos()
            model = focoos.get_local_model(model_ref="<model_ref>")
            metrics = model.end2end_benchmark(iterations=10, size=640)

            # Access latency metrics
            print(f"FPS: {metrics.fps}")
            print(f"Mean latency: {metrics.mean} ms")
            print(f"Engine: {metrics.engine}")
            print(f"Device: {metrics.device}")
            print(f"Input size: {metrics.im_size}x{metrics.im_size}")
            ```
        """
        if size is None:
            size = self.model_info.im_size
        if isinstance(size, int):
            size = (size, size)

        device = get_device_name()
        if self.runtime.__class__.__name__ == "ONNXRuntime":
            active_provider = self.runtime.active_provider or "cpu"  # type: ignore
            engine = f"onnx.{active_provider}"
            if active_provider in ["CPUExecutionProvider"]:
                device = get_cpu_name()
        else:
            engine = "torchscript"
            device = get_device_name()
        logger.info(f"⏱️ Benchmarking End-to-End latency on {device}, size: {size}x{size}..")

        np_input = (255 * np.random.random((size[0], size[1], 3))).astype(np.uint8)

        durations = []
        for step in range(iterations + 5):
            start = perf_counter()
            self.infer(np_input)
            end = perf_counter()

            if step >= 5:  # Skip first 5 iterations
                durations.append((end - start) * 1000)

        durations = np.array(durations)

        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=engine,
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size[0],  # FIXME: this is a hack to get the im_size as int, assuming it's a square
            device=device,
        )
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics
