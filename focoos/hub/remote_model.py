"""
RemoteModel Module

This module provides a class to manage remote models in the Focoos ecosystem. It supports
various functionalities including model training, deployment, inference, and monitoring.

Classes:
    RemoteModel: A class for interacting with remote models, managing their lifecycle,
                 and performing inference.

Methods:
    __init__: Initializes the RemoteModel instance.
    get_info: Retrieves model metadata.
    train: Initiates model training.
    train_info: Retrieves training status.
    train_logs: Retrieves training logs.
    metrics: Retrieves model metrics.
"""

import os
import time
from dataclasses import asdict
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from focoos.hub.api_client import ApiClient
from focoos.ports import (
    ArtifactName,
    FocoosDet,
    FocoosDetections,
    HubSyncLocalTraining,
    Metrics,
    ModelStatus,
    RemoteModelInfo,
    TrainingInfo,
)
from focoos.utils.logger import get_logger
from focoos.utils.metrics import MetricsVisualizer, parse_metrics

logger = get_logger()


class RemoteModel:
    """
    Represents a remote model in the Focoos platform.

    Attributes:
        model_ref (str): Reference ID for the model.
        api_client (ApiClient): Client for making HTTP requests.
        model_info (RemoteModelInfo): Model information of the model.
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
        self.model_info: RemoteModelInfo = self.get_info()

        logger.info(
            f"[RemoteModel]: ref: {self.model_ref} name: {self.model_info.name} description: {self.model_info.description} status: {self.model_info.status}"
        )

    @property
    def ref(self) -> str:
        return self.model_ref

    def get_info(self) -> RemoteModelInfo:
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
        self.metadata = RemoteModelInfo(**res.json())
        return self.metadata

    def sync_local_training_job(
        self, local_training_info: HubSyncLocalTraining, dir: str, upload_artifacts: Optional[List[ArtifactName]] = None
    ) -> None:
        if not os.path.exists(os.path.join(dir, ArtifactName.INFO)):
            logger.warning(f"Model info not found in {dir}")
            raise ValueError(f"Model info not found in {dir}")
        metrics = parse_metrics(os.path.join(dir, ArtifactName.METRICS))
        local_training_info.metrics = metrics
        logger.debug(
            f"[Syncing Training] iter: {metrics.iterations} {self.metadata.name} status: {local_training_info.status} ref: {self.model_ref}"
        )

        ## Update metrics
        res = self.api_client.patch(
            f"models/{self.model_ref}/sync-local-training",
            data=asdict(local_training_info),
        )
        if res.status_code != 200:
            logger.error(f"Failed to sync local training: {res.status_code} {res.text}")
            raise ValueError(f"Failed to sync local training: {res.status_code} {res.text}")
        if upload_artifacts:
            for artifact in [ArtifactName.METRICS, ArtifactName.WEIGHTS]:
                file_path = os.path.join(dir, artifact.value)
                if os.path.isfile(file_path):
                    try:
                        self._upload_model_artifact(file_path)
                    except Exception:
                        logger.error(f"Failed to upload artifact: {artifact.value}")
                        pass

    def _upload_model_artifact(self, path: str) -> None:
        """
        Uploads an model artifact to the Focoos platform.
        """
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        file_ext = os.path.splitext(path)[1]
        if file_ext not in [".pt", ".onnx", ".pth", ".json", ".txt"]:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        file_name = os.path.basename(path)
        file_size = os.path.getsize(path)
        file_size_mb = file_size / (1024 * 1024)
        logger.debug(f"🔗 Requesting upload url for {file_name} of size {file_size_mb:.2f} MB")
        presigned_url = self.api_client.post(
            f"models/{self.model_ref}/generate-upload-url",
            data={"file_size_bytes": file_size, "file_name": file_name},
        )
        if presigned_url.status_code != 200:
            raise ValueError(f"Failed to generate upload url: {presigned_url.status_code} {presigned_url.text}")
        presigned_url = presigned_url.json()
        fields = {k: v for k, v in presigned_url["fields"].items()}
        logger.info(f"📤 Uploading file {file_name} to HUB, size: {file_size_mb:.2f} MB")
        fields["file"] = (file_name, open(path, "rb"))
        res = self.api_client.external_post(
            presigned_url["url"],
            files=fields,
            data=presigned_url["fields"],
        )
        if res.status_code not in [200, 201, 204]:
            raise ValueError(f"Failed to upload model artifact: {res.status_code} {res.text}")
        logger.info(f"✅ Model artifact {file_name} uploaded to HUB.")

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
            logger.error(f"Failed to get train info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get train info: {res.status_code} {res.text}")
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
        return Metrics(**{k: v for k, v in res.json().items() if k in Metrics.__dataclass_fields__})

    def __call__(
        self, image: Union[str, Path, np.ndarray, bytes, Image.Image], threshold: float = 0.5
    ) -> FocoosDetections:
        return self.infer(image, threshold)

    def infer(
        self,
        image: Union[str, Path, np.ndarray, bytes, Image.Image],
        threshold: float = 0.5,
    ) -> FocoosDetections:
        """
        Perform inference on the provided image using the remote model.

        This method sends an image to the remote model for inference and retrieves the detection results.

        Args:
            image (Union[str, Path, np.ndarray, bytes]): The image to infer on, which can be a file path,
                a string representing the path, a NumPy array, or raw bytes.
            threshold (float, optional): The confidence threshold for detections. Defaults to 0.5.
                Detections with confidence scores below this threshold will be discarded.

        Returns:
            FocoosDetections: The detection results including class IDs, confidence scores, bounding boxes,
                and segmentation masks (if applicable).

        Raises:
            FileNotFoundError: If the provided image file path is invalid.
            ValueError: If the inference request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            model = focoos.get_remote_model("my-model")
            results = model.infer("image.jpg", threshold=0.5)

            # Print detection results
            for det in results.detections:
                print(f"Found {det.label} with confidence {det.conf:.2f}")
                print(f"Bounding box: {det.bbox}")
                if det.mask:
                    print("Instance segmentation mask included")
            ```
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

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
        files = {"file": ("image", image_bytes, "image/jpeg")}
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

            return detections
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
