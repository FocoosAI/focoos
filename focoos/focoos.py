import os
from typing import Optional

import requests
from supervision import Detections
from tqdm import tqdm

from focoos.cloud_model import CloudModel
from focoos.ports import (
    DatasetMetadata,
    FocoosEnvHostUrl,
    ModelMetadata,
    ModelPreview,
    ModelStatus,
)
from focoos.utils.logger import get_logger, setup_logging
from focoos.utils.system import HttpClient


class Focoos:
    def __init__(
        self,
        api_key: str = os.getenv("FOCOOS_API_KEY"),
        host_url: FocoosEnvHostUrl = FocoosEnvHostUrl.DEV,
    ):
        self.logger = setup_logging()
        self.api_key = api_key
        if not self.api_key:
            self.logger.error("API key is required 🤖")
            raise ValueError("API key is required 🤖")
        self.http_client = HttpClient(api_key, host_url.value)
        self.user_info = self._get_user_info()
        self.logger.info(
            f"Currently logged as: {self.user_info['email']} environment: {host_url}"
        )

    def _get_user_info(self):
        res = self.http_client.get("user/")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to get user info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get user info: {res.status_code} {res.text}")

    def get_model_info(self, model_name: str) -> ModelMetadata:
        res = self.http_client.get(f"models/{model_name}")
        if res.status_code == 200:
            return ModelMetadata.from_json(res.json())
        else:
            self.logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")

    def list_models(self) -> list[ModelPreview]:
        res = self.http_client.get(f"models/")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to list models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list models: {res.status_code} {res.text}")

    def list_focoos_models(self):
        res = self.http_client.get(f"models/focoos-models")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to list focoos models: {res.status_code} {res.text}"
            )
            raise ValueError(
                f"Failed to list focoos models: {res.status_code} {res.text}"
            )

    def get_model(self, model_ref: str) -> CloudModel:
        return CloudModel(model_ref, self.http_client)

    def new_model(
        self, name: str, focoos_model: str, description: str
    ) -> Optional[CloudModel]:
        res = self.http_client.post(
            f"models/",
            data={
                "name": name,
                "focoos_model": focoos_model,
                "description": description,
            },
        )
        print(res.json())
        if res.status_code in [200, 201]:
            return CloudModel(res.json()["ref"], self.http_client)
        else:
            self.logger.warning(
                f"Failed to create new model: {res.status_code} {res.text}"
            )
            return None

    def list_shared_datasets(self) -> list[DatasetMetadata]:
        res = self.http_client.get(f"datasets/shared")
        if res.status_code == 200:
            return [DatasetMetadata.from_json(dataset) for dataset in res.json()]
        else:
            self.logger.error(f"Failed to list datasets: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list datasets: {res.status_code} {res.text}")
