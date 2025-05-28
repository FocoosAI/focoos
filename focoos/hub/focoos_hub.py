"""
Focoos Module

This module provides a Python interface for interacting with Focoos APIs,
allowing users to manage machine learning models and datasets in the Focoos ecosystem.
The module supports operations such as retrieving model metadata, downloading models,
and listing shared datasets.

Classes:
    FocoosHUB: Main class to interface with Focoos APIs.

Exceptions:
    ValueError: Raised for invalid API responses or missing parameters.
"""

import os
from dataclasses import asdict
from typing import Optional

from focoos.config import FOCOOS_CONFIG
from focoos.hub.api_client import ApiClient
from focoos.hub.remote_dataset import RemoteDataset
from focoos.hub.remote_model import RemoteModel
from focoos.ports import (
    MODELS_DIR,
    ArtifactName,
    DatasetPreview,
    ModelInfo,
    ModelPreview,
    RemoteModelInfo,
    User,
)
from focoos.utils.logger import get_logger

logger = get_logger("HUB")


class FocoosHUB:
    """
    Main class to interface with Focoos APIs.

    This class provides methods to interact with Focoos-hosted models and datasets.
    It supports functionalities such as listing models, retrieving model metadata,
    downloading models, and creating new models.

    Attributes:
        api_key (str): The API key for authentication.
        api_client (ApiClient): HTTP client for making API requests.
        user_info (User): Information about the currently authenticated user.
        host_url (str): Base URL for the Focoos API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host_url: Optional[str] = None,
    ):
        """
        Initializes the FocoosHUB client.

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
            user_info (User): Information about the authenticated user retrieved from the API.
            host_url (str): The base URL used for API requests.

        Logs:
            - Error if the API key or host URL is missing.
            - Info about the authenticated user and environment upon successful initialization.

        Example:
            ```python
            from focoos import FocoosHUB

            focoos = FocoosHUB()
            ```
        """
        self.api_key = api_key or FOCOOS_CONFIG.focoos_api_key
        if not self.api_key:
            logger.error("API key is required 🤖")
            raise ValueError("API key is required 🤖")

        self.host_url = host_url or FOCOOS_CONFIG.default_host_url

        self.api_client = ApiClient(api_key=self.api_key, host_url=self.host_url)
        self.user_info = self.get_user_info()
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
            from focoos import FocoosHUB

            focoos = FocoosHUB()
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

    def get_model_info(self, model_ref: str) -> RemoteModelInfo:
        """
        Retrieves metadata for a specific model.

        Args:
            model_ref (str): Reference identifier for the model.

        Returns:
            RemoteModelInfo: Metadata of the specified model.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import FocoosHUB

            focoos = FocoosHUB()
            model_info = focoos.get_model_info(model_ref="user-or-fai-model-ref")
            ```
        """
        res = self.api_client.get(f"models/{model_ref}")
        if res.status_code != 200:
            logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")
        return RemoteModelInfo.from_json(res.json())

    def list_remote_models(self) -> list[ModelPreview]:
        """
        Lists all models owned by the user.

        Returns:
            list[ModelPreview]: List of model previews.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import FocoosHUB

            focoos = FocoosHUB()
            models = focoos.list_remote_models()
            ```
        """
        res = self.api_client.get("models/")
        if res.status_code != 200:
            logger.error(f"Failed to list models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list models: {res.status_code} {res.text}")
        return [ModelPreview.from_json(r) for r in res.json()]

    def get_remote_model(self, model_ref: str) -> RemoteModel:
        """
        Retrieves a remote model instance for cloud-based inference.

        Args:
            model_ref (str): Reference identifier for the model.

        Returns:
            RemoteModel: The remote model instance configured for cloud-based inference.

        Example:
            ```python
            from focoos import FocoosHUB

            focoos = FocoosHUB()
            model = focoos.get_remote_model(model_ref="fai-model-ref")
            results = model.infer("image.jpg", threshold=0.5)  # inference is remote!
            ```
        """
        return RemoteModel(model_ref, self.api_client)

    def list_shared_datasets(self) -> list[DatasetPreview]:
        """
        Lists datasets shared with the user by others.

        Returns:
            list[DatasetPreview]: List of shared datasets.

        Raises:
            ValueError: If the API request fails.

        Example:
            ```python
            from focoos import FocoosHUB

            focoos = FocoosHUB()
            shared_datasets = focoos.list_shared_datasets()
            ```
        """
        res = self.api_client.get("datasets/shared")
        if res.status_code != 200:
            logger.error(f"Failed to list datasets: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list datasets: {res.status_code} {res.text}")
        return [DatasetPreview.from_json(dataset) for dataset in res.json()]

    def download_model_pth(self, model_ref: str, skip_if_exists: bool = True) -> str:
        """
        Downloads a model from the Focoos API.

        Args:
            model_ref (str): Reference identifier for the model.
            skip_if_exists (bool): If True, skips the download if the model file already exists.
                Defaults to True.

        Returns:
            str: Path to the downloaded model file.

        Raises:
            ValueError: If the API request fails or the download fails.
        """
        model_dir = os.path.join(MODELS_DIR, model_ref)
        model_pth_path = os.path.join(model_dir, ArtifactName.WEIGHTS)
        if os.path.exists(model_pth_path) and skip_if_exists:
            logger.info("📥 Model already downloaded")
            return model_pth_path
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        ## download model metadata
        res = self.api_client.get(f"models/{model_ref}/download?format=pth")
        if res.status_code != 200:
            logger.error(f"Failed to retrieve download url for model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to retrieve download url for model: {res.status_code} {res.text}")

        download_data = res.json()

        download_uri = download_data["download_uri"]
        ## download model from Focoos Cloud
        logger.debug(f"Model URI: {download_uri}")
        logger.info("📥 Downloading model from Focoos Cloud.. ")
        try:
            model_pth_path = self.api_client.download_ext_file(download_uri, model_dir, skip_if_exists=skip_if_exists)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise ValueError(f"Failed to download model: {e}")
        if model_pth_path is None:
            logger.error(f"Failed to download model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to download model: {res.status_code} {res.text}")

        return model_pth_path

    def list_remote_datasets(self, include_shared: bool = False) -> list[DatasetPreview]:
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
            from focoos import FocoosHUB

            focoos = FocoosHUB()

            # List only user's datasets
            datasets = focoos.list_remote_datasets()

            # List user's datasets and shared datasets
            all_datasets = focoos.list_remote_datasets(include_shared=True)

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

    def get_remote_dataset(self, ref: str) -> RemoteDataset:
        """
        Retrieves a remote dataset by its reference ID.

        Args:
            ref (str): The reference ID of the dataset to retrieve.

        Returns:
            RemoteDataset: A RemoteDataset instance for the specified reference.

        Example:
            ```python
            from focoos import FocoosHUB

            focoos = FocoosHUB()
            dataset = focoos.get_remote_dataset(ref="my-dataset-ref")
            ```
        """
        return RemoteDataset(ref, self.api_client)

    def new_model(self, model_info: ModelInfo) -> RemoteModel:
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
            "models/local-model",
            data={
                "name": model_info.name,
                "focoos_model": model_info.focoos_model,
                "description": model_info.description,
                "config": model_info.config if model_info.config else {},
                "task": model_info.task,
                "classes": model_info.classes,
                "im_size": model_info.im_size,
                "train_args": asdict(model_info.train_args) if model_info.train_args else None,
                "focoos_version": model_info.focoos_version,
            },
        )
        if res.status_code in [200, 201]:
            return RemoteModel(res.json()["ref"], self.api_client)
        if res.status_code == 409:
            logger.warning(f"Model already exists: {model_info.name}")
            raise ValueError(f"Failed to create new model: {res.status_code} {res.text}")
        else:
            logger.warning(f"Failed to create new model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to create new model: {res.status_code} {res.text}")
