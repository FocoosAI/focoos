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
import pathlib
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional, Tuple, Union

import numpy as np
import supervision as sv
from PIL import Image

from focoos.config import FOCOOS_CONFIG
from focoos.infer.runtimes.base import BaseRuntime
from focoos.infer.runtimes.load_runtime import load_runtime
from focoos.ports import (
    FocoosDetections,
    InferLatency,
    LatencyMetrics,
    ModelExtension,
    ModelInfo,
    RuntimeType,
)
from focoos.processor.processor_manager import ProcessorManager
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_device_name, get_device_type
from focoos.utils.vision import (
    annotate_image,
    image_loader,
)

logger = get_logger("InferModel")


class InferModel:
    def __init__(
        self,
        model_path: Union[str, Path],
        runtime_type: Optional[RuntimeType] = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
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
        model_extension = pathlib.Path(model_path).suffix
        runtime_type = runtime_type or FOCOOS_CONFIG.runtime_type
        runtime_extension = ModelExtension.from_runtime_type(runtime_type)

        if not model_extension == f".{runtime_extension.value}":
            raise ValueError(
                f"Model extension .{model_extension} mismatch with runtime type: {runtime_type} that expects .{runtime_extension.value}"
            )
        self.device: Literal["cuda", "cpu"]
        if device == "auto":
            self.device = get_device_type()
        elif runtime_type == RuntimeType.ONNX_CPU:
            self.device = "cpu"
        else:
            self.device = device
        # Set model directory and path
        self.model_path = model_path
        self.model_dir: Union[str, Path] = os.path.dirname(str(model_path))
        # self.model_path = os.path.join(model_path, f"model.{extension.value}")
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
            self.processor = ProcessorManager.get_processor(
                self.model_info.model_family, model_config, self.model_info.im_size
            ).eval()
        except Exception as e:
            logger.error(f"Error creating model config: {e}")
            raise e

        # Initialize annotation utilities
        self.label_annotator = sv.LabelAnnotator(text_padding=10, border_radius=10)
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

        # Load runtime for inference
        self.runtime: BaseRuntime = load_runtime(
            runtime_type=runtime_type,
            model_path=str(self.model_path),
            model_info=self.model_info,
            warmup_iter=FOCOOS_CONFIG.warmup_iter,
            device=self.device,
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
        self, image: Union[bytes, str, Path, np.ndarray, Image.Image], threshold: float = 0.5, annotate: bool = False
    ) -> FocoosDetections:
        return self.infer(image, threshold, annotate)

    def infer(
        self,
        image: Union[bytes, str, Path, np.ndarray, Image.Image],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> FocoosDetections:
        """
        Perform inference on an input image and optionally return an annotated result.

        This method processes the input image, runs it through the model, and returns the detections.
        Optionally, it can also annotate the image with the detection results.

        Args:
            image: The input image to run inference on. Accepts a file path, bytes, PIL Image, or numpy array.
            threshold: Minimum confidence score for a detection to be included in the results. Default is 0.5.
            annotate: If True, annotate the image with detection results and include it in the output.

        Returns:
            FocoosDetections: An object containing the detection results, optional annotated image, and latency metrics.

        Raises:
            AssertionError: If the model runtime is not initialized.

        Usage:
            This method is intended for users who want to obtain detection results from a local model,
            with optional annotation for visualization or further processing.
        """
        assert self.runtime is not None, "Model is not deployed (locally)"

        t0 = perf_counter()
        im = image_loader(image)
        t1 = perf_counter()
        tensors, _ = self.processor.preprocess(inputs=im, device=self.device)
        # logger.debug(f"Input image size: {im.shape}")
        t2 = perf_counter()

        raw_detections = self.runtime(tensors)

        t3 = perf_counter()
        detections = self.processor.export_postprocess(
            raw_detections, im, threshold=threshold, class_names=self.model_info.classes
        )
        t4 = perf_counter()
        if annotate:
            skeleton = self.model_info.config.get("skeleton", None)
            detections[0].image = annotate_image(
                im,
                detections[0],
                task=self.model_info.task,
                classes=self.model_info.classes,
                keypoints_skeleton=skeleton,
            )
        t5 = perf_counter()

        res = detections[0]  #!TODO  check for batching
        res.latency = InferLatency(
            imload=round(t1 - t0, 3),
            preprocess=round(t2 - t1, 3),
            inference=round(t3 - t2, 3),
            postprocess=round(t4 - t3, 3),
            annotate=round(t5 - t4, 3) if annotate else None,
        )

        res.infer_print()
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

        # Normalize size to tuple format
        if isinstance(size, int):
            size_tuple = (size, size)
            size_str = f"{size}x{size}"
        else:
            size_tuple = size
            size_str = f"{size[0]}x{size[1]}"

        logger.info(f"⏱️ Benchmarking End-to-End latency on {device}, size: {size_str}..")

        np_input = (255 * np.random.random((size_tuple[0], size_tuple[1], 3))).astype(np.uint8)

        durations = []
        for step in range(iterations + 5):
            start = perf_counter()
            self.infer(np_input)
            end = perf_counter()

            if step >= 5:  # Skip first 5 iterations
                durations.append((end - start) * 1000)

        durations = np.array(durations)

        # For LatencyMetrics.im_size (int), use height (first dimension) as representative value
        # This maintains backward compatibility while supporting non-square images
        im_size_repr = size_tuple[0] if isinstance(size, tuple) and size_tuple[0] != size_tuple[1] else size_tuple[0]
        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=engine,
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=im_size_repr,
            device=device,
        )
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics
