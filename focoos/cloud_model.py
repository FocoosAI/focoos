import os
import time
from pathlib import Path
from typing import Union

from supervision import Detections

from focoos.ports import DeploymentMode, ModelMetadata, ModelStatus
from focoos.utils.logger import get_logger
from focoos.utils.system import HttpClient


class CloudModel:
    def __init__(self, model_ref: str, http_client: HttpClient):
        self.model_ref = model_ref
        self.logger = get_logger()
        self.http_client = http_client
        self.max_deploy_wait = 10
        self.metadata: ModelMetadata = None
        self.info()
        self.logger.info(
            f"[RemoteModel]: ref: {self.model_ref} name: {self.metadata.name} description: {self.metadata.description} status: {self.metadata.status}"
        )

    def info(self) -> ModelMetadata:
        res = self.http_client.get(f"models/{self.model_ref}")
        if res.status_code == 200:
            self.metadata = ModelMetadata(**res.json())
            return self.metadata
        else:
            self.logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")

    def deploy(
        self, deployment_mode: DeploymentMode = DeploymentMode.REMOTE, wait: bool = True
    ):
        if deployment_mode != DeploymentMode.REMOTE:
            raise ValueError("Only remote deployment is supported at the moment")

        if self.metadata.status not in [
            ModelStatus.DEPLOYED,
            ModelStatus.TRAINING_COMPLETED,
        ]:
            raise ValueError(
                f"Model {self.model_ref} is not in a valid state to be deployed. Current status: {self.metadata.status}, expected: {ModelStatus.TRAINING_COMPLETED}"
            )

        if self.metadata.status == ModelStatus.DEPLOYED:
            deployment_info = self._deployment_info()
            self.logger.debug(
                f"Model {self.model_ref} is already deployed, deployment info: {deployment_info}"
            )
            return deployment_info

        self.logger.info(
            f"🚀 Deploying model {self.model_ref} to inference endpoint... this might take a while."
        )
        res = self.http_client.post(f"models/{self.model_ref}/deploy")
        if res.status_code in [200, 201, 409]:
            if res.status_code == 409:
                self.logger.info(f"Status code 409, model is already deployed")

            if wait:
                for i in range(self.max_deploy_wait):
                    self.logger.info(
                        f"⏱️ Waiting for model {self.model_ref} to be ready... {i+1} of {self.max_deploy_wait}"
                    )
                    if self._deployment_info()["status"] == "READY":
                        self.logger.info(
                            f"✅ Model {self.model_ref} deployed successfully"
                        )
                        return
                    time.sleep(1 + i)
                self.logger.error(
                    f"Model {self.model_ref} deployment timed out after {self.max_deploy_wait} attempts."
                )
                raise ValueError(
                    f"Model {self.model_ref} deployment timed out after {self.max_deploy_wait} attempts."
                )
            return res.json()
        else:
            self.logger.error(f"Failed to deploy model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to deploy model: {res.status_code} {res.text}")

    def unload(self):
        res = self.http_client.delete(f"models/{self.model_ref}/deploy")
        if res.status_code in [200, 204, 409]:
            return res.json()
        else:
            self.logger.error(f"Failed to unload model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to unload model: {res.status_code} {res.text}")

    def train_logs(self) -> list[str]:
        res = self.http_client.get(f"models/{self.model_ref}/train-logs")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to get train logs: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get train logs: {res.status_code} {res.text}")

    def _deployment_info(self):
        res = self.http_client.get(f"models/{self.model_ref}/deploy")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to get deployment info: {res.status_code} {res.text}"
            )
            raise ValueError(
                f"Failed to get deployment info: {res.status_code} {res.text}"
            )

    def infer(self, image_path: Union[str, Path], threshold: float = 0.5) -> Detections:
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        files = {"file": open(image_path, "rb")}
        t0 = time.time()
        res = self.http_client.post(
            f"models/{self.model_ref}/inference?confidence_threshold={threshold}",
            files=files,
        )
        t1 = time.time()
        if res.status_code == 200:
            self.logger.debug(f"Inference time: {t1-t0:.3f} seconds")
            return res.json()
        else:
            self.logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")

    def infer_preview(
        self, image_path: Union[str, Path], threshold: float = 0.5
    ) -> Detections:
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        files = {"file": open(image_path, "rb")}
        t0 = time.time()
        res = self.http_client.post(
            f"models/{self.model_ref}/inference?confidence_threshold={threshold}",
            extra_headers={"Accept": "image/jpeg"},
            files=files,
        )
        t1 = time.time()
        if res.status_code == 200:
            self.logger.debug(f"Inference time: {t1-t0:.3f} seconds")
            return res.content
        else:
            self.logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")
