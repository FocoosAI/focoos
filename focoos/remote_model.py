import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from supervision import BoxAnnotator, Detections, LabelAnnotator, MaskAnnotator
from tqdm import tqdm

from focoos.config import FocoosConfig
from focoos.ports import (
    DeploymentMode,
    FocoosDet,
    FocoosDetections,
    FocoosTask,
    Hyperparameters,
    LatencyMetrics,
    ModelMetadata,
    ModelStatus,
    OnnxEngineOpts,
    TrainInstance,
)
from focoos.runtime import ONNXRuntime
from focoos.utils.logger import get_logger
from focoos.utils.system import HttpClient
from focoos.utils.vision import (
    focoos_detections_to_supervision,
    image_loader,
    image_preprocess,
    scale_detections,
    sv_to_focoos_detections,
)

config = FocoosConfig()

logger = get_logger()


class RemoteModel:
    def __init__(self, model_ref: str, http_client: HttpClient):
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
        res = self.http_client.get(f"models/{self.model_ref}")
        if res.status_code == 200:
            return ModelMetadata(**res.json())
        else:
            logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")

    def train(
        self,
        dataset_ref: str,
        hyperparameters: Hyperparameters,
        anyma_version: str = "anyma-sagemaker-cu12-torch22-0111",
        instance_type: TrainInstance = TrainInstance.ML_G4DN_XLARGE,
        volume_size: int = 50,
        max_runtime_in_seconds: int = 36000,
    ):
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
        if res.status_code == 200:
            return res.json()
        else:
            logger.warning(f"Failed to train model: {res.status_code} {res.text}")
            return None

    def train_status(self):
        res = self.http_client.get(f"models/{self.model_ref}/train/status")
        if res.status_code == 200:
            return res.json()
        else:
            logger.error(f"Failed to get train status: {res.status_code} {res.text}")
            raise ValueError(
                f"Failed to get train status: {res.status_code} {res.text}"
            )

    def deploy(self, wait: bool = True):
        self.get_info()
        if self.metadata.status not in [
            ModelStatus.DEPLOYED,
            ModelStatus.TRAINING_COMPLETED,
        ]:
            raise ValueError(
                f"Model {self.model_ref} is not in a valid state to be deployed. Current status: {self.metadata.status}, expected: {ModelStatus.TRAINING_COMPLETED}"
            )
        if self.metadata.status == ModelStatus.DEPLOYED:
            deployment_info = self._deployment_info()
            logger.debug(
                f"Model {self.model_ref} is already deployed, deployment info: {deployment_info}"
            )
            return deployment_info

        logger.info(
            f"🚀 Deploying model {self.model_ref} to inference endpoint... this might take a while."
        )
        res = self.http_client.post(f"models/{self.model_ref}/deploy")
        if res.status_code in [200, 201, 409]:
            if res.status_code == 409:
                logger.info(f"Status code 409, model is already deployed")

            if wait:
                for i in range(self.max_deploy_wait):
                    logger.info(
                        f"⏱️ Waiting for model {self.model_ref} to be ready... {i+1} of {self.max_deploy_wait}"
                    )
                    if self._deployment_info()["status"] == "READY":
                        logger.info(f"✅ Model {self.model_ref} deployed successfully")
                        return
                    time.sleep(1 + i)
                logger.error(
                    f"Model {self.model_ref} deployment timed out after {self.max_deploy_wait} attempts."
                )
                raise ValueError(
                    f"Model {self.model_ref} deployment timed out after {self.max_deploy_wait} attempts."
                )
            return res.json()
        else:
            logger.error(f"Failed to deploy model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to deploy model: {res.status_code} {res.text}")

    def unload(self):
        res = self.http_client.delete(f"models/{self.model_ref}/deploy")
        if res.status_code in [200, 204, 409]:
            return res.json()
        else:
            logger.error(f"Failed to unload model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to unload model: {res.status_code} {res.text}")

    def train_logs(self) -> list[str]:
        res = self.http_client.get(f"models/{self.model_ref}/train/logs")
        if res.status_code == 200:
            return res.json()
        else:
            logger.warning(f"Failed to get train logs: {res.status_code} {res.text}")
            return []

    def _deployment_info(self):
        res = self.http_client.get(f"models/{self.model_ref}/deploy")
        if res.status_code == 200:
            return res.json()
        else:
            logger.error(f"Failed to get deployment info: {res.status_code} {res.text}")
            raise ValueError(
                f"Failed to get deployment info: {res.status_code} {res.text}"
            )

    def _annotate(self, im: np.ndarray, detections: Detections) -> np.ndarray:
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
        elif self.metadata.task == FocoosTask.SEMSEG:
            annotated_im = self.mask_annotator.annotate(
                scene=im.copy(), detections=detections
            )
        return annotated_im

    def infer(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> Tuple[FocoosDetections, Optional[np.ndarray]]:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        files = {"file": open(image_path, "rb")}
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
                im0 = image_loader(image_path)
                sv_detections = focoos_detections_to_supervision(detections)
                preview = self._annotate(im0, sv_detections)
            return detections, preview
        else:
            logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")
