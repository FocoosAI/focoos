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
        runtime_type: Optional[RuntimeTypes] = None,
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
            label_annotator (LabelAnnotator): Utility for adding labels to the output,
                initialized with text padding and border radius.
            box_annotator (BoxAnnotator): Utility for annotating bounding boxes.
            mask_annotator (MaskAnnotator): Utility for annotating masks.
            runtime (ONNXRuntime): Inference runtime initialized with the specified runtime type,
                model path, metadata, and warmup iterations.

        The method verifies the existence of the model directory, reads the model metadata,
        and initializes the runtime for inference using the provided runtime type. Annotation
        utilities are also prepared for visualizing model outputs.
        """
        runtime_type = runtime_type or FOCOOS_CONFIG.runtime_type
        if not runtime_type:
            raise ValueError("Runtime type is required for local model")

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
                This can be a byte array, file path, or a PIL Image object, or a NumPy array representing the image.
            threshold (float, optional): The confidence threshold for detections. Defaults to 0.5.
                Detections with confidence scores below this threshold will be discarded.
            annotate (bool, optional): Whether to annotate the image with detection results. Defaults to False.
                If set to True, the method will return the image with bounding boxes or segmentation masks.

        Returns:
            Tuple[FocoosDetections, Optional[np.ndarray]]: A tuple containing:
                - `FocoosDetections`: The detections from the inference, represented as a custom object (`FocoosDetections`).
                This includes the details of the detected objects such as class, confidence score, and bounding box (if applicable).
                - `Optional[np.ndarray]`: The annotated image, if `annotate=True`.
                This will be a NumPy array representation of the image with drawn bounding boxes or segmentation masks.
                If `annotate=False`, this value will be `None`.

        Raises:
            ValueError: If the model is not deployed locally (i.e., `self.runtime` is `None`).
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
