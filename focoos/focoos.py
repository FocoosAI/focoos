import json
import os
from pathlib import Path
from typing import Optional

import requests
from supervision import Detections
from tqdm import tqdm

from focoos.cloud_model import CloudModel
from focoos.config import FocoosCfg
from focoos.ports import (
    DatasetMetadata,
    FocoosEnvHostUrl,
    ModelMetadata,
    ModelPreview,
    ModelStatus,
    NewModel,
)
from focoos.utils.logger import get_logger, setup_logging
from focoos.utils.system import HttpClient

cfg = FocoosCfg()


class Focoos:
    def __init__(
        self,
        api_key: str = cfg.focoos_api_key,
        host_url: FocoosEnvHostUrl = cfg.host_url,
        user_dir: Path = cfg.user_dir,
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
        self.user_dir = Path(user_dir)
        if not self.user_dir.exists():
            self.user_dir.mkdir(parents=True)

    def _download_model(self, model_ref: str) -> str:
        dir = self.user_dir / model_ref
        model_path = dir / "model.onnx"
        metadata_path = dir / "focoos_metadata.json"
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            self.logger.info(f"📥 Model already exists: {dir}")
            return dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        res = self.http_client.get(f"models/{model_ref}/download?format=onnx")
        if res.status_code == 200:
            metadata = self.get_model_info(model_ref)
            model_uri = res.json()

            with open(metadata_path, "w") as f:
                json.dump(metadata.model_dump_json(), f)

            self.logger.info(f"Metadata: {metadata_path}")

            self.logger.info(f"📥 Downloading model from Focoos Cloud.. ")
            response = requests.get(model_uri, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                self.logger.info(f"📥 Size: {total_size / (1024**2):.2f} MB")
                with open(model_path, "wb") as f, tqdm(
                    desc=str(model_path).split("/")[-1],
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
                self.logger.info(f"📥 File downloaded: {model_path}")
                return metadata

        self.logger.error(f"Failed to download model: {res.status_code} {res.text}")
        raise ValueError(f"Failed to download model: {res.status_code} {res.text}")

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

    def new_model(self, model: NewModel) -> Optional[CloudModel]:
        res = self.http_client.post(f"models/", data=model.model_dump())
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
