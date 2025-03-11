"""
RemoteModel Module

This module provides a class to manage remote models in the Focoos ecosystem. It supports
various functionalities including model training, deployment, inference, and monitoring.

Classes:
    RemoteModel: A class for interacting with remote models, managing their lifecycle,
                 and performing inference.


Modules:
    ApiClient: Handles HTTP requests.
    logger: Logging utility.
    BoxAnnotator, LabelAnnotator, MaskAnnotator: Annotation tools for visualizing
                 detections and segmentation tasks.
    FocoosDet, FocoosDetections: Classes for representing and managing detections.
    FocoosTask: Enum for defining supported tasks (e.g., DETECTION, SEMSEG).
    Hyperparameters: Structure for training configuration parameters.
    ModelMetadata: Contains metadata for the model.
    ModelStatus: Enum for representing the current status of the model.
    TrainInstance: Enum for defining available training instances.
    image_loader: Utility function for loading images.
"""

import os
import time
from pathlib import Path
from time import sleep
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv

from focoos.ports import (
    FocoosDet,
    FocoosDetections,
    FocoosTask,
    Hyperparameters,
    Metrics,
    ModelMetadata,
    ModelStatus,
    TrainingInfo,
    TrainInstance,
)
from focoos.utils.api_client import ApiClient
from focoos.utils.logger import get_logger
from focoos.utils.metrics import MetricsVisualizer
from focoos.utils.vision import fai_detections_to_sv, image_loader

logger = get_logger()


class RemoteModel:
    """
    Represents a remote model in the Focoos platform.

    Attributes:
        model_ref (str): Reference ID for the model.
        api_client (ApiClient): Client for making HTTP requests.
        max_deploy_wait (int): Maximum wait time for model deployment.
        metadata (ModelMetadata): Metadata of the model.
        label_annotator (LabelAnnotator): Annotator for adding labels to images.
        box_annotator (sv.BoxAnnotator): Annotator for drawing bounding boxes.
        mask_annotator (sv.MaskAnnotator): Annotator for drawing masks on images.
    """

    def __init__(
        self,
        model_ref: str,
        api_client: ApiClient,
    ):
        """
        Initialize the RemoteModel instance.

        Args:
            model_ref (str): Reference ID for the model.
            api_client (ApiClient): HTTP client instance for communication.

        Raises:
            ValueError: If model metadata retrieval fails.
        """
        self.model_ref = model_ref
        self.api_client = api_client
        self.metadata: ModelMetadata = self.get_info()

        self.label_annotator = sv.LabelAnnotator(text_padding=10, border_radius=10)
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        logger.info(
            f"[RemoteModel]: ref: {self.model_ref} name: {self.metadata.name} description: {self.metadata.description} status: {self.metadata.status}"
        )

    def get_info(self) -> ModelMetadata:
        """
        Retrieve model metadata.

        Returns:
            ModelMetadata: Metadata of the model.

        Raises:
            ValueError: If the request fails.

        Example:
            ```python
            from focoos import Focoos, RemoteModel

            focoos = Focoos()
            model = focoos.get_remote_model(model_ref="<model_ref>")
            model_info = model.get_info()
            ```
        """
        res = self.api_client.get(f"models/{self.model_ref}")
        if res.status_code != 200:
            logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")
        self.metadata = ModelMetadata(**res.json())
        return self.metadata

    def train(
        self,
        dataset_ref: str,
        hyperparameters: Hyperparameters,
        instance_type: TrainInstance = TrainInstance.ML_G4DN_XLARGE,
        volume_size: int = 50,
        max_runtime_in_seconds: int = 36000,
    ) -> dict | None:
        """
        Initiate the training of a remote model on the Focoos platform.

        This method sends a request to the Focoos platform to start the training process for the model
        referenced by `self.model_ref`. It requires a dataset reference and hyperparameters for training,
        as well as optional configuration options for the instance type, volume size, and runtime.

        Args:
            dataset_ref (str): The reference ID of the dataset to be used for training.
            hyperparameters (Hyperparameters): A structure containing the hyperparameters for the training process.
            anyma_version (str, optional): The version of Anyma to use for training. Defaults to "anyma-sagemaker-cu12-torch22-0111".
            instance_type (TrainInstance, optional): The type of training instance to use. Defaults to TrainInstance.ML_G4DN_XLARGE.
            volume_size (int, optional): The size of the disk volume (in GB) for the training instance. Defaults to 50.
            max_runtime_in_seconds (int, optional): The maximum runtime for training in seconds. Defaults to 36000.

        Returns:
            dict: A dictionary containing the response from the training initiation request. The content depends on the Focoos platform's response.

        Raises:
            ValueError: If the request to start training fails (e.g., due to incorrect parameters or server issues).
        """
        res = self.api_client.post(
            f"models/{self.model_ref}/train",
            data={
                "dataset_ref": dataset_ref,
                "instance_type": instance_type,
                "volume_size": volume_size,
                "max_runtime_in_seconds": max_runtime_in_seconds,
                "hyperparameters": hyperparameters.model_dump(),
            },
        )
        if res.status_code != 200:
            logger.warning(f"Failed to train model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to train model: {res.status_code} {res.text}")
        return res.json()

    def train_info(self) -> Optional[TrainingInfo]:
        """
        Retrieve the current status of the model training.

        Sends a request to check the training status of the model referenced by `self.model_ref`.

        Returns:
            dict: A dictionary containing the training status information.

        Raises:
            ValueError: If the request to get training status fails.
        """
        res = self.api_client.get(f"models/{self.model_ref}/train/status")
        if res.status_code != 200:
            logger.error(f"Failed to get train status: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get train status: {res.status_code} {res.text}")
        return TrainingInfo(**res.json())

    def train_logs(self) -> list[str]:
        """
        Retrieve the training logs for the model.

        This method sends a request to fetch the logs of the model's training process. If the request
        is successful (status code 200), it returns the logs as a list of strings. If the request fails,
        it logs a warning and returns an empty list.

        Returns:
            list[str]: A list of training logs as strings.

        Raises:
            None: Returns an empty list if the request fails.
        """
        res = self.api_client.get(f"models/{self.model_ref}/train/logs")
        if res.status_code != 200:
            logger.warning(f"Failed to get train logs: {res.status_code} {res.text}")
            return []
        return res.json()

    def metrics(self) -> Metrics:  # noqa: F821
        """
        Retrieve the metrics of the model.

        This method sends a request to fetch the metrics of the model identified by `model_ref`.
        If the request is successful (status code 200), it returns the metrics as a `Metrics` object.
        If the request fails, it logs a warning and returns an empty `Metrics` object.

        Returns:
            Metrics: An object containing the metrics of the model.

        Raises:
            None: Returns an empty `Metrics` object if the request fails.
        """
        res = self.api_client.get(f"models/{self.model_ref}/metrics")
        if res.status_code != 200:
            logger.warning(f"Failed to get metrics: {res.status_code} {res.text}")
            return Metrics()  # noqa: F821
        return Metrics(**res.json())

    def _annotate(self, im: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotate an image with detection results.

        This method adds visual annotations to the provided image based on the model's detection results.
        It handles different tasks (e.g., object detection, semantic segmentation, instance segmentation)
        and uses the corresponding annotator (bounding box, label, or mask) to draw on the image.

        Args:
            im (np.ndarray): The image to be annotated, represented as a NumPy array.
            detections (sv.Detections): The detection results to be annotated, including class IDs and confidence scores.

        Returns:
            np.ndarray: The annotated image as a NumPy array.
        """

        if len(detections.xyxy) == 0:
            logger.warning("No detections found, skipping annotation")
            return im
        classes = self.metadata.classes
        if classes is not None:
            labels = [
                f"{classes[int(class_id)]}: {confid * 100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)
            ]
        else:
            labels = [
                f"{str(class_id)}: {confid * 100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)
            ]
        if self.metadata.task == FocoosTask.DETECTION:
            annotated_im = self.box_annotator.annotate(scene=im.copy(), detections=detections)

            annotated_im = self.label_annotator.annotate(scene=annotated_im, detections=detections, labels=labels)
        elif self.metadata.task in [
            FocoosTask.SEMSEG,
            FocoosTask.INSTANCE_SEGMENTATION,
        ]:
            annotated_im = self.mask_annotator.annotate(scene=im.copy(), detections=detections)
        return annotated_im

    def infer(
        self,
        image: Union[str, Path, np.ndarray, bytes],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> Tuple[FocoosDetections, Optional[np.ndarray]]:
        """
        Perform inference on the provided image using the remote model.

        This method sends an image to the remote model for inference and retrieves the detection results.
        Optionally, it can annotate the image with the detection results.

        Args:
            image (Union[str, Path, np.ndarray, bytes]): The image to infer on, which can be a file path,
                a string representing the path, a NumPy array, or raw bytes.
            threshold (float, optional): The confidence threshold for detections. Defaults to 0.5.
                Detections with confidence scores below this threshold will be discarded.
            annotate (bool, optional): Whether to annotate the image with the detection results. Defaults to False.
                If set to True, the method will return the image with bounding boxes or segmentation masks.

        Returns:
            Tuple[FocoosDetections, Optional[np.ndarray]]:
                - FocoosDetections: The detection results including class IDs, confidence scores, bounding boxes,
                  and segmentation masks (if applicable).
                - Optional[np.ndarray]: The annotated image if `annotate` is True, else None.
                  This will be a NumPy array representation of the image with drawn bounding boxes or segmentation masks.

        Raises:
            FileNotFoundError: If the provided image file path is invalid.
            ValueError: If the inference request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()

            model = focoos.get_remote_model("my-model")
            results, annotated_image = model.infer("image.jpg", threshold=0.5, annotate=True)

            # Print detection results
            for det in results.detections:
                print(f"Found {det.label} with confidence {det.conf:.2f}")
                print(f"Bounding box: {det.bbox}")
                if det.mask:
                    print("Instance segmentation mask included")
            ```
        """
        image_bytes = None
        if isinstance(image, str) or isinstance(image, Path):
            if not os.path.exists(image):
                logger.error(f"Image file not found: {image}")
                raise FileNotFoundError(f"Image file not found: {image}")
            image_bytes = open(image, "rb").read()
        elif isinstance(image, np.ndarray):
            _, buffer = cv2.imencode(".jpg", image)
            image_bytes = buffer.tobytes()
        else:
            image_bytes = image
        files = {"file": image_bytes}
        t0 = time.time()
        res = self.api_client.post(
            f"models/{self.model_ref}/inference?confidence_threshold={threshold}",
            files=files,
        )
        t1 = time.time()
        if res.status_code == 200:
            detections = FocoosDetections(
                detections=[FocoosDet.from_json(d) for d in res.json().get("detections", [])],
                latency=res.json().get("latency", None),
            )
            logger.debug(
                f"Found {len(detections.detections)} detections. Inference Request time: {(t1 - t0) * 1000:.0f}ms"
            )
            preview = None
            if annotate:
                im0 = image_loader(image)
                sv_detections = fai_detections_to_sv(detections, im0.shape[:-1])
                preview = self._annotate(im0, sv_detections)
            return detections, preview
        else:
            logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")

    def notebook_monitor_train(self, interval: int = 30, plot_metrics: bool = False, max_runtime: int = 36000) -> None:
        """
        Monitor the training process in a Jupyter notebook and display metrics.

        Periodically checks the training status and displays metrics in a notebook cell.
        Clears previous output to maintain a clean view.

        Args:
            interval (int): Time between status checks in seconds. Must be 30-240. Default: 30
            plot_metrics (bool): Whether to plot metrics graphs. Default: False
            max_runtime (int): Maximum monitoring time in seconds. Default: 36000 (10 hours)

        Returns:
            None
        """
        from IPython.display import clear_output

        if not 30 <= interval <= 240:
            raise ValueError("Interval must be between 30 and 240 seconds")

        last_update = self.get_info().updated_at
        start_time = time.time()
        status_history = []

        while True:
            # Get current status
            model_info = self.get_info()
            status = model_info.status

            # Clear and display status
            clear_output(wait=True)
            status_msg = f"[Live Monitor {self.metadata.name}] {status.value}"
            status_history.append(status_msg)
            for msg in status_history:
                logger.info(msg)

            # Show metrics if training completed
            if status == ModelStatus.TRAINING_COMPLETED:
                metrics = self.metrics()
                if metrics.best_valid_metric:
                    logger.info(f"Best Checkpoint (iter: {metrics.best_valid_metric.get('iteration', 'N/A')}):")
                    for k, v in metrics.best_valid_metric.items():
                        logger.info(f"  {k}: {v}")
                    visualizer = MetricsVisualizer(metrics)
                    visualizer.log_metrics()
                    if plot_metrics:
                        visualizer.notebook_plot_training_metrics()

            # Update metrics during training
            if status == ModelStatus.TRAINING_RUNNING and model_info.updated_at > last_update:
                last_update = model_info.updated_at
                metrics = self.metrics()
                visualizer = MetricsVisualizer(metrics)
                visualizer.log_metrics()
                if plot_metrics:
                    visualizer.notebook_plot_training_metrics()

            # Check exit conditions
            if status not in [ModelStatus.CREATED, ModelStatus.TRAINING_RUNNING, ModelStatus.TRAINING_STARTING]:
                return

            if time.time() - start_time > max_runtime:
                logger.warning(f"Monitoring exceeded {max_runtime} seconds limit")
                return

            sleep(interval)

    def stop_training(self) -> None:
        """
        Stop the training process of the model.

        This method sends a request to stop the training of the model identified by `model_ref`.
        If the request fails, an error is logged and a `ValueError` is raised.

        Raises:
            ValueError: If the stop training request fails.

        Logs:
            - Error message if the request to stop training fails, including the status code and response text.

        Returns:
            None: This method does not return any value.
        """
        res = self.api_client.delete(f"models/{self.model_ref}/train")
        if res.status_code != 200:
            logger.error(f"Failed to get stop training: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get stop training: {res.status_code} {res.text}")

    def delete_model(self) -> None:
        """
        Delete the model from the system.

        This method sends a request to delete the model identified by `model_ref`.
        If the request fails or the status code is not 204 (No Content), an error is logged
        and a `ValueError` is raised.

        Raises:
            ValueError: If the delete model request fails or does not return a 204 status code.

        Logs:
            - Error message if the request to delete the model fails, including the status code and response text.

        Returns:
            None: This method does not return any value.
        """
        res = self.api_client.delete(f"models/{self.model_ref}")
        if res.status_code != 204:
            logger.error(f"Failed to delete model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to delete model: {res.status_code} {res.text}")
