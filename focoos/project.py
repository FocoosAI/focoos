from typing import Optional

from focoos.ports import ModelMetadata, ModelPreview, NewModel, ProjectMetadata
from focoos.utils.logger import get_logger
from focoos.utils.system import HttpClient


class Project:
    def __init__(
        self, project_ref: str, http_client: HttpClient, project_info: dict = {}
    ):
        self.project_ref = project_ref
        self.metadata = ProjectMetadata(**project_info)
        self.http_client = http_client
        self.logger = get_logger()

    def list_models(self) -> list[ModelPreview]:
        res = self.http_client.get(f"projects/{self.project_ref}/models")
        if res.status_code == 200:
            return [ModelPreview(**model) for model in res.json()]
        else:
            self.logger.error(f"Failed to list models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list models: {res.status_code} {res.text}")

    def add_model(self, new_model: NewModel) -> Optional[ModelMetadata]:
        res = self.http_client.post(
            f"projects/{self.project_ref}/models", data=new_model.model_dump()
        )
        if res.status_code == 200:
            return ModelMetadata(**res.json())
        else:
            self.logger.warning(f"Failed to add model: {res.status_code} {res.text}")
            return None
