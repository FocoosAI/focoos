"""
Focoos Module

This module provides a Python interface for interacting with Focoos APIs,
allowing users to manage machine learning models and datasets in the Focoos ecosystem.
The module supports operations such as retrieving model metadata, downloading models,
and listing shared datasets.

Classes:
    Focoos: Main class to interface with Focoos APIs.

Exceptions:
    ValueError: Raised for invalid API responses or missing parameters.
"""

import os
from typing import Optional, Union

from focoos.config import FOCOOS_CONFIG
from focoos.local_model import LocalModel
from focoos.ports import (
    DatasetLayout,
    DatasetPreview,
    ModelFormat,
    ModelInfo,
    ModelNotFound,
    ModelPreview,
    RuntimeTypes,
    Task,
    User,
)
from focoos.remote_dataset import RemoteDataset
from focoos.remote_model import RemoteModel
from focoos.utils.api_client import ApiClient
from focoos.utils.logger import setup_logging

logger = setup_logging()


class Focoos:
    """
    Main class to interface with Focoos APIs.

    This class provides methods to interact with Focoos-hosted models and datasets.
    It supports functionalities such as listing models, retrieving model metadata,
    downloading models, and creating new models.

    Attributes:
        api_key (str): The API key for authentication.
        api_client (ApiClient): HTTP client for making API requests.
        user_info (dict): Information about the currently authenticated user.
        cache_dir (str): Local directory for caching downloaded models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host_url: Optional[str] = None,
    ):
        """
        Initializes the Focoos API client.

        This client provides authenticated access to the Focoos API, enabling various operations
        through the configured HTTP client. It retrieves user information upon initialization and
        logs the environment details.

        Args:
            api_key (Optional[str]): API key for authentication. Defaults to the `focoos_api_key`
                specified in the FOCOOS_CONFIG.
            host_url (Optional[str]): Base URL for the Focoos API. Defaults to the `default_host_url`
                specified in the FOCOOS_CONFIG.

        Raises:
            ValueError: If the API key is not provided, or if the host URL is not specified in the
                arguments or the configuration.

        Attributes:
            api_key (str): The API key used for authentication.
            api_client (ApiClient): An HTTP client instance configured with the API key and host URL.
            user_info (dict): Information about the authenticated user retrieved from the API.
            cache_dir (str): Path to the cache directory used by the client.

        Logs:
            - Error if the API key or host URL is missing.
            - Info about the authenticated user and environment upon successful initialization.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            ```
        """
        self.api_key = api_key or FOCOOS_CONFIG.focoos_api_key
        if not self.api_key:
            logger.error("API key is required 🤖")
            raise ValueError("API key is required 🤖")

        self.host_url = host_url or FOCOOS_CONFIG.default_host_url

        self.api_client = ApiClient(api_key=self.api_key, host_url=self.host_url)
        self.user_info = self.get_user_info()
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "focoos")
        logger.info(f"Currently logged as: {self.user_info.email} environment: {self.host_url}")

    def get_user_info(self) -> User:
        """
        Retrieves information about the authenticated user.

        Returns:
            User: User object containing account information and usage quotas.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            user_info = focoos.get_user_info()

            # Access user info fields
            print(f"Email: {user_info.email}")
            print(f"Created at: {user_info.created_at}")
            print(f"Updated at: {user_info.updated_at}")
            print(f"Company: {user_info.company}")
            print(f"API key: {user_info.api_key.key}")

            # Access quotas
            quotas = user_info.quotas
            print(f"Total inferences: {quotas.total_inferences}")
            print(f"Max inferences: {quotas.max_inferences}")
            print(f"Used storage (GB): {quotas.used_storage_gb}")
            print(f"Max storage (GB): {quotas.max_storage_gb}")
            print(f"Active training jobs: {quotas.active_training_jobs}")
            print(f"Max active training jobs: {quotas.max_active_training_jobs}")
            print(f"Used MLG4DNXLarge training jobs hours: {quotas.used_mlg4dnxlarge_training_jobs_hours}")
            print(f"Max MLG4DNXLarge training jobs hours: {quotas.max_mlg4dnxlarge_training_jobs_hours}")
            ```
        """
        res = self.api_client.get("user/")
        if res.status_code != 200:
            logger.error(f"Failed to get user info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get user info: {res.status_code} {res.text}")
        return User.from_json(res.json())

    def get_model_info(self, model_ref: str) -> ModelInfo:
        """
        Retrieves metadata for a specific model.

        Args:
            model_ref (str): Name of the model.

        Returns:
            ModelMetadata: Metadata of the specified model.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            model_info = focoos.get_model_info(model_ref=<user-or-fai-model-ref>)
            ```
        """
        res = self.api_client.get(f"models/{model_ref}")
        if res.status_code != 200:
            logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")
        return ModelInfo.from_json(res.json())

    def list_models(self) -> list[ModelPreview]:
        """
        Lists all User Models.

        Returns:
            list[ModelPreview]: List of model previews.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            models = focoos.list_models()
            ```
        """
        res = self.api_client.get("models/")
        if res.status_code != 200:
            logger.error(f"Failed to list models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list models: {res.status_code} {res.text}")
        return [ModelPreview.from_json(r) for r in res.json()]

    def list_focoos_models(self) -> list[ModelPreview]:
        """
        Lists FAI shared models.

        Returns:
            list[ModelPreview]: List of Focoos models.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            focoos_models = focoos.list_focoos_models()
            ```
        """
        res = self.api_client.get("models/focoos-models")
        if res.status_code != 200:
            logger.error(f"Failed to list focoos models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list focoos models: {res.status_code} {res.text}")
        return [ModelPreview.from_json(r) for r in res.json()]

    def get_local_model(
        self,
        model_ref: str,
        runtime_type: Optional[RuntimeTypes] = RuntimeTypes.ONNX_CUDA32,
    ) -> LocalModel:
        """
        Retrieves a local model for the specified reference.

        Downloads the model if it does not already exist in the local cache.

        Args:
            model_ref (str): Reference identifier for the model.
            runtime_type (Optional[RuntimeTypes]): Runtime type for the model. Defaults to the
                `runtime_type` specified in FOCOOS_CONFIG.

        Returns:
            LocalModel: An instance of the local model.

        Raises:
            ValueError: If the runtime type is not specified.

        Notes:
            The model is cached in the directory specified by `self.cache_dir`.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            model = focoos.get_local_model(model_ref=<user-or-fai-model-ref>)
            results, annotated_image = model.infer("image.jpg", threshold=0.5, annotate=True) # inference is local!
            ```
        """
        runtime_type = runtime_type or FOCOOS_CONFIG.runtime_type
        model_dir = os.path.join(self.cache_dir, model_ref)
        format = ModelFormat.from_runtime_type(runtime_type)
        if not os.path.exists(os.path.join(model_dir, f"model.{format.value}")):
            self._download_model(
                model_ref,
                format=format,
            )
        return LocalModel(model_dir, runtime_type)

    def get_remote_model(self, model_ref: str) -> RemoteModel:
        """
        Retrieves a remote model instance.

        Args:
            model_ref (str): Reference name of the model.

        Returns:
            RemoteModel: The remote model instance.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            model = focoos.get_remote_model(model_ref=<fai-model-ref>)
            results, annotated_image = model.infer("image.jpg", threshold=0.5, annotate=True) # inference is remote!
            ```
        """
        return RemoteModel(model_ref, self.api_client)

    def new_model(self, name: str, focoos_model: str, description: str) -> RemoteModel:
        """
        Creates a new model in the Focoos platform.

        Args:
            name (str): Name of the new model.
            focoos_model (str): Reference to the base Focoos model.
            description (str): Description of the new model.

        Returns:
            Optional[RemoteModel]: The created model instance, or None if creation fails.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            model = focoos.new_model(name="my-model", focoos_model="fai-model-ref", description="my-model-description")
            ```
        """
        res = self.api_client.post(
            "models/",
            data={
                "name": name,
                "focoos_model": focoos_model,
                "description": description,
            },
        )
        if res.status_code in [200, 201]:
            return RemoteModel(res.json()["ref"], self.api_client)
        if res.status_code == 409:
            logger.warning(f"Model already exists: {name}")
            return self.get_model_by_name(name, remote=True)
        logger.warning(f"Failed to create new model: {res.status_code} {res.text}")

    def list_shared_datasets(self) -> list[DatasetPreview]:
        """
        Lists datasets shared with the user.

        Returns:
            list[DatasetPreview]: List of shared datasets.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            datasets = focoos.list_shared_datasets()
            ```
        """
        res = self.api_client.get("datasets/shared")
        if res.status_code != 200:
            logger.error(f"Failed to list datasets: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list datasets: {res.status_code} {res.text}")
        return [DatasetPreview.from_json(dataset) for dataset in res.json()]

    def _download_model(self, model_ref: str, format: ModelFormat = ModelFormat.ONNX) -> str:
        """
        Downloads a model from the Focoos API.

        Args:
            model_ref (str): Reference name of the model.

        Returns:
            str: Path to the downloaded model.

        Raises:
            ValueError: If the API request fails or the download fails.
        """
        model_dir = os.path.join(self.cache_dir, model_ref)
        model_path = os.path.join(model_dir, f"model.{format.value}")
        metadata_path = os.path.join(model_dir, "focoos_metadata.json")
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            logger.info("📥 Model already downloaded")
            return model_path
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        ## download model metadata
        res = self.api_client.get(f"models/{model_ref}/download?format={format.value}")
        if res.status_code != 200:
            logger.error(f"Failed to retrieve download url for model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to retrieve download url for model: {res.status_code} {res.text}")

        download_data = res.json()

        download_uri = download_data["download_uri"]

        ## download model from Focoos Cloud
        logger.debug(f"Model URI: {download_uri}")
        logger.info("📥 Downloading model from Focoos Cloud.. ")
        try:
            model_path = self.api_client.download_file(download_uri, model_dir)
            metadata = ModelInfo.from_json(download_data["model_metadata"])
            with open(metadata_path, "w") as f:
                f.write(metadata.model_dump_json())
            logger.debug(f"Dumped metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise ValueError(f"Failed to download model: {e}")
        if model_path is None:
            logger.error(f"Failed to download model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to download model: {res.status_code} {res.text}")

        return model_path

    def get_model_by_name(self, name: str, remote: bool = True) -> Union[RemoteModel, LocalModel]:
        """
        Retrieves a model by its name.

        Args:
            name (str): Name of the model.
            remote (bool): If True, retrieve as a RemoteModel. Otherwise, as a LocalModel. Defaults to True.

        Returns:
            Optional[Union[RemoteModel, LocalModel]]: The model instance if found, or None otherwise.
        """
        models = self.list_models()
        name_lower = name.lower()
        for model in models:
            if name_lower == model.name.lower():
                if remote:
                    return self.get_remote_model(model.ref)
                else:
                    return self.get_local_model(model.ref)
        raise ModelNotFound(f"Model not found: {name}")

    def list_datasets(self, include_shared: bool = False) -> list[DatasetPreview]:
        """
        Lists all datasets available to the user.

        This method retrieves all datasets owned by the user and optionally includes
        shared datasets as well.

        Args:
            include_shared (bool): If True, includes datasets shared with the user.
                Defaults to False.

        Returns:
            list[DatasetPreview]: A list of DatasetPreview objects representing the available datasets.

        Raises:
            ValueError: If the API request to list datasets fails.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()

            # List only user's datasets
            datasets = focoos.list_datasets()

            # List user's datasets and shared datasets
            all_datasets = focoos.list_datasets(include_shared=True)

            for dataset in all_datasets:
                print(f"Dataset: {dataset.name}, Task: {dataset.task}")
            ```
        """
        res = self.api_client.get("datasets/")
        if res.status_code != 200:
            logger.error(f"Failed to list datasets: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list datasets: {res.status_code} {res.text}")
        datasets = [DatasetPreview.from_json(r) for r in res.json()]
        if include_shared:
            res = self.api_client.get("datasets/shared")
            if res.status_code != 200:
                logger.error(f"Failed to list datasets: {res.status_code} {res.text}")
                raise ValueError(f"Failed to list datasets: {res.status_code} {res.text}")
            datasets.extend([DatasetPreview.from_json(sh_dataset) for sh_dataset in res.json()])
        return datasets

    def add_remote_dataset(self, name: str, description: str, layout: DatasetLayout, task: Task) -> RemoteDataset:
        """
        Creates a new user dataset with the specified parameters.

        Args:
            name (str): The name of the dataset.
            description (str): A description of the dataset.
            layout (DatasetLayout): The layout structure of the dataset.
            task (FocoosTask): The task type associated with the dataset.

        Returns:
            RemoteDataset: A RemoteDataset instance representing the newly created dataset.

        Raises:
            ValueError: If the dataset creation fails due to API errors.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            dataset = focoos.add_remote_dataset(name="my-dataset", description="my-dataset-description", layout=DatasetLayout.ROBOFLOW_COCO, task=FocoosTask.DETECTION)
            ```
        """
        res = self.api_client.post(
            "datasets/", data={"name": name, "description": description, "layout": layout.value, "task": task.value}
        )
        if res.status_code != 200:
            logger.error(f"Failed to add dataset: {res.status_code} {res.text}")
            raise ValueError(f"Failed to add dataset: {res.status_code} {res.text}")
        logger.info(f"Remote Dataset created: {res.json()['ref']}")
        return RemoteDataset(res.json()["ref"], self.api_client)

    def get_remote_dataset(self, ref: str) -> RemoteDataset:
        """
        Retrieves a remote dataset by its reference ID.

        Args:
            ref (str): The reference ID of the dataset to retrieve.

        Returns:
            RemoteDataset: A RemoteDataset instance for the specified reference.

        Example:
            ```python
            from focoos import Focoos

            focoos = Focoos()
            dataset = focoos.get_remote_dataset(ref="my-dataset-ref")
            ```
        """
        return RemoteDataset(ref, self.api_client)
