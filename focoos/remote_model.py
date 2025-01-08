"""
RemoteModel Module

This module provides a class to manage remote models in the Focoos ecosystem. It supports
various functionalities including model training, deployment, inference, and monitoring.

Classes:
    RemoteModel: A class for interacting with remote models, managing their lifecycle,
                 and performing inference.


Modules:
    HttpClient: Handles HTTP requests.
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
    focoos_detections_to_supervision: Converter for Focoos detections to supervision format.
"""

import os
import time
from pathlib import Path
from time import sleep
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from supervision import BoxAnnotator, Detections, LabelAnnotator, MaskAnnotator

from focoos.ports import (
    FocoosDet,
    FocoosDetections,
    FocoosTask,
    Hyperparameters,
    ModelMetadata,
    TrainInstance,
)
from focoos.utils.logger import get_logger
from focoos.utils.system import HttpClient
from focoos.utils.vision import focoos_detections_to_supervision, image_loader

logger = get_logger()


class RemoteModel:
    """
    Represents a remote model in the Focoos platform.

    Attributes:
        model_ref (str): Reference ID for the model.
        http_client (HttpClient): Client for making HTTP requests.
        max_deploy_wait (int): Maximum wait time for model deployment.
        metadata (ModelMetadata): Metadata of the model.
        label_annotator (LabelAnnotator): Annotator for adding labels to images.
        box_annotator (BoxAnnotator): Annotator for drawing bounding boxes.
        mask_annotator (MaskAnnotator): Annotator for drawing masks on images.
    """

    def __init__(self, model_ref: str, http_client: HttpClient):
        """
        Initialize the RemoteModel instance.

        Args:
            model_ref (str): Reference ID for the model.
            http_client (HttpClient): HTTP client instance for communication.

        Raises:
            ValueError: If model metadata retrieval fails.
        """
        self.model_ref = model_ref
        self.http_client = http_client
        self.max_deploy_wait = 10
        self.metadata: ModelMetadata = self.get_info()

        self.label_annotator = LabelAnnotator(text_padding=10, border_radius=10)
        self.box_annotator = BoxAnnotator()
        self.mask_annotator = MaskAnnotator()
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
        """
        res = self.http_client.get(f"models/{self.model_ref}")
        if res.status_code != 200:
            logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")
        self.metadata = ModelMetadata(**res.json())
        return self.metadata

    def train(
        self,
        dataset_ref: str,
        hyperparameters: Hyperparameters,
        anyma_version: str = "anyma-sagemaker-cu12-torch22-0111",
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
        res = self.http_client.post(
            f"models/{self.model_ref}/train",
            data={
                "dataset_ref": dataset_ref,
                "anyma_version": anyma_version,
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

    def train_status(self) -> dict | None:
        """
        Retrieve the current status of the model training.

        Sends a request to check the training status of the model referenced by `self.model_ref`.

        Returns:
            dict: A dictionary containing the training status information.

        Raises:
            ValueError: If the request to get training status fails.
        """
        res = self.http_client.get(f"models/{self.model_ref}/train/status")
        if res.status_code != 200:
            logger.error(f"Failed to get train status: {res.status_code} {res.text}")
            raise ValueError(
                f"Failed to get train status: {res.status_code} {res.text}"
            )
        return res.json()

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
        res = self.http_client.get(f"models/{self.model_ref}/train/logs")
        if res.status_code != 200:
            logger.warning(f"Failed to get train logs: {res.status_code} {res.text}")
            return []
        return res.json()

    def _annotate(self, im: np.ndarray, detections: Detections) -> np.ndarray:
        """
        Annotate an image with detection results.

        This method adds visual annotations to the provided image based on the model's detection results.
        It handles different tasks (e.g., object detection, semantic segmentation, instance segmentation)
        and uses the corresponding annotator (bounding box, label, or mask) to draw on the image.

        Args:
            im (np.ndarray): The image to be annotated, represented as a NumPy array.
            detections (Detections): The detection results to be annotated, including class IDs and confidence scores.

        Returns:
            np.ndarray: The annotated image as a NumPy array.
        """
        classes = self.metadata.classes
        if classes is not None:
            labels = [
                f"{classes[int(class_id)]}: {confid*100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)
            ]
        else:
            labels = [
                f"{str(class_id)}: {confid*100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)
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
        image: Union[str, Path, np.ndarray, bytes],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> Tuple[FocoosDetections, Optional[np.ndarray]]:
        """
        Perform inference on the provided image using the remote model.

        This method sends an image to the remote model for inference and retrieves the detection results.
        Optionally, it can annotate the image with the detection results.

        Args:
            image (Union[str, Path, bytes]): The image to infer on, which can be a file path, a string representing the path, or raw bytes.
            threshold (float, optional): The confidence threshold for detections. Defaults to 0.5.
            annotate (bool, optional): Whether to annotate the image with the detection results. Defaults to False.

        Returns:
            Tuple[FocoosDetections, Optional[np.ndarray]]:
                - FocoosDetections: The detection results including class IDs, confidence scores, etc.
                - Optional[np.ndarray]: The annotated image if `annotate` is True, else None.

        Raises:
            FileNotFoundError: If the provided image file path is invalid.
            ValueError: If the inference request fails.
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
        res = self.http_client.post(
            f"models/{self.model_ref}/inference?confidence_threshold={threshold}",
            files=files,
        )
        t1 = time.time()
        if res.status_code == 200:
            logger.debug(f"Inference time: {t1-t0:.3f} seconds")
            detections = FocoosDetections(
                detections=[
                    FocoosDet.from_json(d) for d in res.json().get("detections", [])
                ],
                latency=res.json().get("latency", None),
            )
            preview = None
            if annotate:
                im0 = image_loader(image)
                sv_detections = focoos_detections_to_supervision(detections)
                preview = self._annotate(im0, sv_detections)
            return detections, preview
        else:
            logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")

    def train_metrics(self, period=60) -> dict | None:
        """
        Retrieve training metrics for the model over a specified period.

        This method fetches the training metrics for the remote model, including aggregated values,
        such as average performance metrics over the given period.

        Args:
            period (int, optional): The period (in seconds) for which to fetch the metrics. Defaults to 60.

        Returns:
            Optional[dict]: A dictionary containing the training metrics if the request is successful,
                            or None if the request fails.
        """
        res = self.http_client.get(
            f"models/{self.model_ref}/train/all-metrics?period={period}&aggregation_type=Average"
        )
        if res.status_code != 200:
            logger.warning(f"Failed to get train logs: {res.status_code} {res.text}")
            return None
        return res.json()

    def _log_metrics(self):
        """
        Log the latest training metrics for the model.

        This method retrieves the current training metrics, such as iteration, total loss, and evaluation
        metrics (like mIoU for segmentation tasks or AP50 for detection tasks). It logs the most recent values
        for these metrics, helping monitor the model's training progress.

        The logged metrics depend on the model's task:
            - For segmentation tasks (SEMSEG), the mean Intersection over Union (mIoU) is logged.
            - For detection tasks, the Average Precision at 50% IoU (AP50) is logged.

        Returns:
            None: The method only logs the metrics without returning any value.

        Logs:
            - Iteration number.
            - Total loss value.
            - Relevant evaluation metric (mIoU or AP50).
        """
        metrics = self.train_metrics()
        if metrics:
            iter = (
                metrics["iter"][-1]
                if "iter" in metrics and len(metrics["iter"]) > 0
                else -1
            )
            total_loss = (
                metrics["total_loss"][-1]
                if "total_loss" in metrics and len(metrics["total_loss"]) > 0
                else -1
            )
            if self.metadata.task == FocoosTask.SEMSEG:
                accuracy = (
                    metrics["mIoU"][-1]
                    if "mIoU" in metrics and len(metrics["mIoU"]) > 0
                    else "-"
                )
                eval_metric = "mIoU"
            else:
                accuracy = (
                    metrics["AP50"][-1]
                    if "AP50" in metrics and len(metrics["AP50"]) > 0
                    else "-"
                )
                eval_metric = "AP50"
            logger.info(
                f"Iter {iter:.0f}: Loss {total_loss:.2f}, {eval_metric} {accuracy}"
            )

    def monitor_train(self, update_period=30) -> None:
        """
        Monitor the training process of the model and log its status periodically.

        This method continuously checks the model's training status and logs updates based on the current state.
        It monitors the primary and secondary statuses of the model, and performs the following actions:
        - If the status is "Pending", it logs a waiting message and waits for resources.
        - If the status is "InProgress", it logs the current status and elapsed time, and logs the training metrics if the model is actively training.
        - If the status is "Completed", it logs the final metrics and exits.
        - If the training fails, is stopped, or any unexpected status occurs, it logs the status and exits.

        Args:
            update_period (int, optional): The time (in seconds) to wait between status checks. Default is 30 seconds.

        Returns:
            None: This method does not return any value but logs information about the training process.

        Logs:
            - The current training status, including elapsed time.
            - Training metrics at regular intervals while the model is training.
        """
        completed_status = ["Completed", "Failed", "Stopped"]
        # init to make do-while
        status = {"main_status": "Flag", "secondary_status": "Flag"}
        prev_status = status
        while status["main_status"] not in completed_status:
            prev_status = status
            status = self.train_status()
            elapsed = status.get("elapsed_time", 0)
            # Model at the startup
            if not status["main_status"] or status["main_status"] in ["Pending"]:
                if prev_status["main_status"] != status["main_status"]:
                    logger.info("[0s] Waiting for resources...")
                sleep(update_period)
                continue
            # Training in progress
            if status["main_status"] in ["InProgress"]:
                if prev_status["secondary_status"] != status["secondary_status"]:
                    if status["secondary_status"] in ["Starting", "Pending"]:
                        logger.info(
                            f"[0s] {status['main_status']}: {status['secondary_status']}"
                        )
                    else:
                        logger.info(
                            f"[{elapsed//60}m:{elapsed%60}s] {status['main_status']}: {status['secondary_status']}"
                        )
                if status["secondary_status"] in ["Training"]:
                    self._log_metrics()
                sleep(update_period)
                continue
            if status["main_status"] == "Completed":
                self._log_metrics()
                return
            else:
                logger.info(f"Model is not training, status: {status['main_status']}")
                return

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
        res = self.http_client.delete(f"models/{self.model_ref}/train")
        if res.status_code != 200:
            logger.error(f"Failed to get stop training: {res.status_code} {res.text}")
            raise ValueError(
                f"Failed to get stop training: {res.status_code} {res.text}"
            )

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
        res = self.http_client.delete(f"models/{self.model_ref}")
        if res.status_code != 204:
            logger.error(f"Failed to delete model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to delete model: {res.status_code} {res.text}")
