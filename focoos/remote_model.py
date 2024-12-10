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
        self.metadata = ModelMetadata(**res.json())
        if res.status_code == 200:
            return self.metadata
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

    def train_logs(self) -> list[str]:
        res = self.http_client.get(f"models/{self.model_ref}/train/logs")
        if res.status_code == 200:
            return res.json()
        else:
            logger.warning(f"Failed to get train logs: {res.status_code} {res.text}")
            return []

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

    def train_metrics(self, period=60) -> Optional[dict]:
        res = self.http_client.get(
            f"models/{self.model_ref}/train/all-metrics?period={period}&aggregation_type=Average"
        )
        if res.status_code == 200:
            return res.json()
        else:
            logger.warning(f"Failed to get train logs: {res.status_code} {res.text}")
            return None

    def _log_metrics(self):
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

    def monitor_train(self, update_period=30):
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

    def stop_training(self):
        res = self.http_client.delete(f"models/{self.model_ref}/train")
        if res.status_code != 200:
            logger.error(f"Failed to get stop training: {res.status_code} {res.text}")
            raise ValueError(
                f"Failed to get stop training: {res.status_code} {res.text}"
            )

    def delete_model(self):
        res = self.http_client.delete(f"models/{self.model_ref}")
        if res.status_code != 204:
            logger.error(f"Failed to delete model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to delete model: {res.status_code} {res.text}")
