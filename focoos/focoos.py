from typing import Optional

import requests
from supervision import Detections

from focoos.cloud_model import CloudModel
from focoos.ports import (
    FocoosEnvHostUrl,
    ModelMetadata,
    ModelPreview,
    ModelStatus,
    NewModel,
    ProjectMetadata,
)
from focoos.project import Project
from focoos.utils.logger import get_logger, setup_logging
from focoos.utils.system import HttpClient


class Focoos:
    def __init__(self, api_key: str, host_url: FocoosEnvHostUrl = FocoosEnvHostUrl.DEV):
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

    def list_projects(self) -> list[ProjectMetadata]:
        res = self.http_client.get(f"projects/")
        if res.status_code == 200:
            return [ProjectMetadata(**project) for project in res.json()]
        else:
            self.logger.error(f"Failed to list projects: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list projects: {res.status_code} {res.text}")

    def create_project(
        self,
        project_name: str,
        description: str,
        dataset_url: str,
        dataset_name: str,
        dataset_layout: str,
        task: str = "detection",
    ) -> Optional[Project]:
        dataset = {
            "dataset_url": dataset_url,
            "dataset_name": dataset_name,
            "layout": dataset_layout,
        }
        res = self.http_client.post(
            f"projects/",
            data={
                "name": project_name,
                "description": description,
                "dataset": dataset,
                "task": task,
            },
        )
        if res.status_code in [200, 201]:
            return Project(res.json()["ref"], self.http_client, res.json())
        else:
            self.logger.warning(
                f"Failed to create project: {res.status_code} {res.text}"
            )
            return None

    def get_project(self, project_ref: str) -> Optional[Project]:
        res = self.http_client.get(f"projects/{project_ref}")
        if res.status_code == 200:
            return Project(project_ref, self.http_client, res.json())
        else:
            self.logger.error(
                f"Failed to get project info: {res.status_code} {res.text}"
            )
            return None

    def get_model_info(self, model_name: str):
        res = self.http_client.get(f"models/{model_name}")
        if res.status_code == 200:
            return res.json()
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
