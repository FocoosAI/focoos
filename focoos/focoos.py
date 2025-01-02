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

from tqdm import tqdm

from focoos.config import FOCOOS_CONFIG
from focoos.local_model import LocalModel
from focoos.ports import DatasetMetadata, ModelMetadata, ModelPreview, RuntimeTypes
from focoos.remote_model import RemoteModel
from focoos.utils.logger import setup_logging
from focoos.utils.system import HttpClient

logger = setup_logging()


class Focoos:
    """
    Main class to interface with Focoos APIs.

    This class provides methods to interact with Focoos-hosted models and datasets.
    It supports functionalities such as listing models, retrieving model metadata,
    downloading models, and creating new models.

    Attributes:
        api_key (str): The API key for authentication.
        http_client (HttpClient): HTTP client for making API requests.
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
            http_client (HttpClient): An HTTP client instance configured with the API key and host URL.
            user_info (dict): Information about the authenticated user retrieved from the API.
            cache_dir (str): Path to the cache directory used by the client.

        Logs:
            - Error if the API key or host URL is missing.
            - Info about the authenticated user and environment upon successful initialization.
        """
        self.api_key = api_key or FOCOOS_CONFIG.focoos_api_key
        if not self.api_key:
            logger.error("API key is required 🤖")
            raise ValueError("API key is required 🤖")

        host_url = host_url or FOCOOS_CONFIG.default_host_url

        self.http_client = HttpClient(self.api_key, host_url)
        self.user_info = self._get_user_info()
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "focoos")
        logger.info(
            f"Currently logged as: {self.user_info['email']} environment: {host_url}"
        )

    def _get_user_info(self):
        """
        Retrieves information about the authenticated user.

        Returns:
            dict: Information about the user (e.g., email).

        Raises:
            ValueError: If the API request fails.
        """
        res = self.http_client.get("user/")
        if res.status_code != 200:
            logger.error(f"Failed to get user info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get user info: {res.status_code} {res.text}")
        return res.json()

    def get_model_info(self, model_name: str) -> ModelMetadata:
        """
        Retrieves metadata for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            ModelMetadata: Metadata of the specified model.

        Raises:
            ValueError: If the API request fails.
        """
        res = self.http_client.get(f"models/{model_name}")
        if res.status_code != 200:
            logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")
        return ModelMetadata.from_json(res.json())

    def list_models(self) -> list[ModelPreview]:
        """
        Lists all available models.

        Returns:
            list[ModelPreview]: List of model previews.

        Raises:
            ValueError: If the API request fails.
        """
        res = self.http_client.get("models/")
        if res.status_code != 200:
            logger.error(f"Failed to list models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list models: {res.status_code} {res.text}")
        return [ModelPreview.from_json(r) for r in res.json()]

    def list_focoos_models(self) -> list[ModelPreview]:
        """
        Lists models specific to Focoos.

        Returns:
            list[ModelPreview]: List of Focoos models.

        Raises:
            ValueError: If the API request fails.
        """
        res = self.http_client.get("models/focoos-models")
        if res.status_code != 200:
            logger.error(f"Failed to list focoos models: {res.status_code} {res.text}")
            raise ValueError(
                f"Failed to list focoos models: {res.status_code} {res.text}"
            )
        return [ModelPreview.from_json(r) for r in res.json()]

    def get_local_model(
        self,
        model_ref: str,
        runtime_type: Optional[RuntimeTypes] = None,
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
        """
        runtime_type = runtime_type or FOCOOS_CONFIG.runtime_type
        model_dir = os.path.join(self.cache_dir, model_ref)
        if not os.path.exists(os.path.join(model_dir, "model.onnx")):
            self._download_model(model_ref)
        return LocalModel(model_dir, runtime_type)

    def get_remote_model(self, model_ref: str) -> RemoteModel:
        """
        Retrieves a remote model instance.

        Args:
            model_ref (str): Reference name of the model.

        Returns:
            RemoteModel: The remote model instance.
        """
        return RemoteModel(model_ref, self.http_client)

    def new_model(
        self, name: str, focoos_model: str, description: str
    ) -> Optional[RemoteModel]:
        """
        Creates a new model in the Focoos system.

        Args:
            name (str): Name of the new model.
            focoos_model (str): Reference to the base Focoos model.
            description (str): Description of the new model.

        Returns:
            Optional[RemoteModel]: The created model instance, or None if creation fails.

        Raises:
            ValueError: If the API request fails.
        """
        res = self.http_client.post(
            "models/",
            data={
                "name": name,
                "focoos_model": focoos_model,
                "description": description,
            },
        )
        if res.status_code in [200, 201]:
            return RemoteModel(res.json()["ref"], self.http_client)
        if res.status_code == 409:
            logger.warning(f"Model already exists: {name}")
            return self.get_model_by_name(name, remote=True)
        logger.warning(f"Failed to create new model: {res.status_code} {res.text}")
        return None

    def list_shared_datasets(self) -> list[DatasetMetadata]:
        """
        Lists datasets shared with the user.

        Returns:
            list[DatasetMetadata]: List of shared datasets.

        Raises:
            ValueError: If the API request fails.
        """
        res = self.http_client.get("datasets/shared")
        if res.status_code != 200:
            logger.error(f"Failed to list datasets: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list datasets: {res.status_code} {res.text}")
        return [DatasetMetadata.from_json(dataset) for dataset in res.json()]

    def _download_model(self, model_ref: str) -> str:
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
        model_path = os.path.join(model_dir, "model.onnx")
        metadata_path = os.path.join(model_dir, "focoos_metadata.json")
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            logger.info("📥 Model already downloaded")
            return model_path
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        res = self.http_client.get(f"models/{model_ref}/download?format=onnx")
        if res.status_code != 200:
            logger.error(f"Failed to download model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to download model: {res.status_code} {res.text}")

        download_data = res.json()
        metadata = ModelMetadata.from_json(download_data["model_metadata"])
        with open(metadata_path, "w") as f:
            f.write(metadata.model_dump_json())

        logger.debug(f"Dumped metadata to {metadata_path}")
        download_uri = download_data["download_uri"]
        logger.debug(f"Model URI: {download_uri}")
        logger.info("📥 Downloading model from Focoos Cloud.. ")
        response = self.http_client.get_external_url(download_uri, stream=True)
        if response.status_code != 200:
            logger.error(
                f"Failed to download model: {response.status_code} {response.text}"
            )
            raise ValueError(
                f"Failed to download model: {response.status_code} {response.text}"
            )
        total_size = int(response.headers.get("content-length", 0))
        logger.info(f"📥 Size: {total_size / (1024**2):.2f} MB")
        with (
            open(model_path, "wb") as f,
            tqdm(
                desc=str(model_path).split("/")[-1],
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        logger.info(f"📥 File downloaded: {model_path}")
        return model_path

    def get_dataset_by_name(self, name: str) -> Optional[DatasetMetadata]:
        """
        Retrieves a dataset by its name.

        Args:
            name (str): Name of the dataset.

        Returns:
            Optional[DatasetMetadata]: The dataset metadata if found, or None otherwise.
        """
        datasets = self.list_shared_datasets()
        for dataset in datasets:
            if name.lower() == dataset.name.lower():
                return dataset

    def get_model_by_name(
        self, name: str, remote: bool = True
    ) -> Optional[Union[RemoteModel, LocalModel]]:
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
