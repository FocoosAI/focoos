"""
LocalModel Module

This module provides the `LocalModel` class that allows loading, inference,
and benchmark testing of models in a local environment. It supports detection
and segmentation tasks, and utilizes ONNXRuntime for model execution.

Classes:
    LocalModel: A class for managing and interacting with local models.

Methods:
    __init__: Initializes the LocalModel instance, loading the model, metadata,
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
from PIL import Image
from supervision import BoxAnnotator, Detections, LabelAnnotator, MaskAnnotator

from focoos.config import FOCOOS_CONFIG
from focoos.ports import (
    FocoosDetections,
    FocoosTask,
    LatencyMetrics,
    ModelMetadata,
    RuntimeTypes,
)
from focoos.runtime import ONNXRuntime, get_runtime
from focoos.utils.logger import get_logger
from focoos.utils.vision import (
    image_preprocess,
    scale_detections,
    sv_to_focoos_detections,
)

logger = get_logger(__name__)


class LocalModel:
    def __init__(
        self,
        model_dir: Union[str, Path],
        runtime_type: RuntimeTypes = FOCOOS_CONFIG.runtime_type,
    ):
        """
        Initialize the LocalModel instance.

        Args:
            model_dir (Union[str, Path]): Path to the model directory.
            runtime_type (RuntimeTypes, optional): Type of runtime to use. Defaults to
                                                FOCOOS_CONFIG.runtime_type.

        Raises:
            FileNotFoundError: If the specified model directory does not exist.

        Initializes the model, loads metadata, and prepares the runtime environment
        for inference.
        """
        logger.debug(f"Runtime type: {runtime_type}, Loading model from {model_dir},")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.model_dir: Union[str, Path] = model_dir
        self.metadata: ModelMetadata = self._read_metadata()
        self.model_ref = self.metadata.ref
        self.label_annotator = LabelAnnotator(text_padding=10, border_radius=10)
        self.box_annotator = BoxAnnotator()
        self.mask_annotator = MaskAnnotator()
        self.runtime: ONNXRuntime = get_runtime(
            runtime_type,
            str(os.path.join(model_dir, "model.onnx")),
            self.metadata,
            FOCOOS_CONFIG.warmup_iter,
        )

    def _read_metadata(self) -> ModelMetadata:
        """
        Reads the model metadata from a JSON file.

        Returns:
            ModelMetadata: Metadata for the model.

        Raises:
            FileNotFoundError: If the metadata file does not exist in the model directory.
        """
        metadata_path = os.path.join(self.model_dir, "focoos_metadata.json")
        return ModelMetadata.from_json(metadata_path)

    def _annotate(self, im: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotates the input image with detection or segmentation results.

        Args:
            im (np.ndarray): The input image to annotate.
            detections (Detections): Detected objects or segmented regions.

        Returns:
            np.ndarray: The annotated image with bounding boxes or masks.
        """
        classes = self.metadata.classes
        if classes is not None:
            labels = [
                f"{classes[int(class_id)]}: {confid*100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)  # type: ignore
            ]
        else:
            labels = [
                f"{str(class_id)}: {confid*100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)  # type: ignore
            ]
        if self.metadata.task == FocoosTask.DETECTION:
            annotated_im = self.box_annotator.annotate(
                scene=im.copy(), detections=detections
            )

            annotated_im = self.label_annotator.annotate(
                scene=annotated_im, detections=detections, labels=labels
            )
        elif self.metadata.task in [
            FocoosTask.SEMSEG,
            FocoosTask.INSTANCE_SEGMENTATION,
        ]:
            annotated_im = self.mask_annotator.annotate(
                scene=im.copy(), detections=detections
            )
        return annotated_im

    def infer(
        self,
        image: Union[bytes, str, Path, np.ndarray, Image.Image],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> Tuple[FocoosDetections, Optional[np.ndarray]]:
        """
        Run inference on an input image and optionally annotate the results.

        Args:
            image (Union[bytes, str, Path, np.ndarray, Image.Image]): The input image to infer on.
            threshold (float, optional): The confidence threshold for detections. Defaults to 0.5.
            annotate (bool, optional): Whether to annotate the image with detection results. Defaults to False.

        Returns:
            Tuple[FocoosDetections, Optional[np.ndarray]]: The detections from the inference and the annotated image (if applicable).

        Raises:
            ValueError: If the model is not deployed locally.
        """
        if self.runtime is None:
            raise ValueError("Model is not deployed (locally)")
        resize = None  #!TODO  check for segmentation
        if self.metadata.task == FocoosTask.DETECTION:
            resize = 640 if not self.metadata.im_size else self.metadata.im_size
        logger.debug(f"Resize: {resize}")
        t0 = perf_counter()
        im1, im0 = image_preprocess(image, resize=resize)
        t1 = perf_counter()
        detections = self.runtime(im1.astype(np.float32), threshold)
        t2 = perf_counter()
        if resize:
            detections = scale_detections(
                detections, (resize, resize), (im0.shape[1], im0.shape[0])
            )
        logger.debug(f"Inference time: {t2-t1:.3f} seconds")
        im = None
        if annotate:
            im = self._annotate(im0, detections)

        out = sv_to_focoos_detections(detections, classes=self.metadata.classes)
        t3 = perf_counter()
        out.latency = {
            "inference": round(t2 - t1, 3),
            "preprocess": round(t1 - t0, 3),
            "postprocess": round(t3 - t2, 3),
        }
        return out, im

    def benchmark(self, iterations: int, size: int) -> LatencyMetrics:
        """
        Benchmark the model's inference performance over multiple iterations.

        Args:
            iterations (int): Number of iterations to run for benchmarking.
            size (int): The input size for each benchmark iteration.

        Returns:
            LatencyMetrics: Latency metrics including time taken for inference.
        """
        return self.runtime.benchmark(iterations, size)
