import os
from typing import Optional

from focoos.ports import DatasetPreview, DatasetSpec
from focoos.utils.api_client import ApiClient
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class RemoteDataset:
    """
    A class to manage remote datasets through the Focoos API.

    This class provides functionality to interact with datasets stored remotely,
    including uploading, downloading, and managing dataset data.

    Args:
        ref (str): The reference identifier for the dataset.
        api_client (ApiClient): The API client instance for making requests.

    Attributes:
        ref (str): The dataset reference identifier.
        api_client (ApiClient): The API client instance.
        metadata (DatasetPreview): The dataset metadata.
    """

    def __init__(self, ref: str, api_client: ApiClient):
        self.ref = ref
        self.api_client = api_client
        self.metadata: DatasetPreview = self.get_info()

    def get_info(self) -> DatasetPreview:
        """
        Retrieves the dataset information from the API.

        Returns:
            DatasetPreview: The dataset preview information.
        """
        res = self.api_client.get(f"datasets/{self.ref}")
        return DatasetPreview.from_json(res.json())

    def delete(self):
        """
        Deletes the entire dataset from the remote storage.

        Raises:
            Exception: If the deletion fails.
        """
        try:
            res = self.api_client.delete(f"datasets/{self.ref}")
            res.raise_for_status()
            logger.warning(f"Deleted dataset {self.ref}")
        except Exception as e:
            logger.error(f"Failed to delete dataset {self.ref}: {e}")
            raise e

    def delete_data(self):
        """
        Deletes only the data content of the dataset while preserving metadata.

        Updates the metadata after successful deletion.
        """
        try:
            res = self.api_client.delete(f"datasets/{self.ref}/data")

            res.raise_for_status()
            new_metadata = DatasetPreview.from_json(res.json())
            self.metadata = new_metadata
            logger.warning(f"Deleted dataset data {self.ref}")
        except Exception as e:
            logger.error(f"Failed to delete dataset data {self.ref}: {e}")

    def upload_data(self, path: str) -> Optional[DatasetSpec]:
        """
        Uploads dataset data from a local zip file to the remote storage.

        Args:
            path (str): Local path to the zip file containing dataset data.

        Returns:
            Optional[DatasetSpec]: The dataset specification after successful upload.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a zip file or upload fails.
        """
        if not path.endswith(".zip"):
            raise ValueError("Dataset must be .zip compressed")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        file_name = os.path.basename(path)
        file_size = os.path.getsize(path)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"🔗 Requesting upload url for {file_name} of size {file_size_mb:.2f} MB")
        presigned_url = self.api_client.post(
            f"datasets/{self.ref}/generate-upload-url",
            data={"file_size_bytes": file_size, "file_name": file_name},
        )
        if presigned_url.status_code != 200:
            raise ValueError(f"Failed to generate upload url: {presigned_url.status_code} {presigned_url.text}")
        presigned_url = presigned_url.json()
        fields = {k: v for k, v in presigned_url["fields"].items()}
        logger.info(f"📤 Uploading file {file_name}..")
        fields["file"] = (file_name, open(path, "rb"), "application/zip")

        res = self.api_client.external_post(
            presigned_url["url"],
            files=fields,
            data=presigned_url["fields"],
            stream=True,
        )
        logger.info("✅ Upload file done.")
        if res.status_code not in [200, 201, 204]:
            raise ValueError(f"Failed to upload dataset: {res.status_code} {res.text}")

        logger.info("🔗 Validating dataset..")
        complete_upload = self.api_client.post(
            f"datasets/{self.ref}/complete-upload",
        )
        complete_upload.raise_for_status()
        self.metadata = self.get_info()
        logger.info(f"✅ Dataset validated! => {self.metadata.spec}")
        return self.metadata.spec

    def download_data(self, path: str):
        """
        Downloads the dataset data to a local path.

        Args:
            path (str): Local path where the dataset should be downloaded.

        Returns:
            str: The path where the file was downloaded.

        Raises:
            ValueError: If the download fails.
        """
        res = self.api_client.get(f"datasets/{self.ref}/download")
        if res.status_code != 200:
            raise ValueError(f"Failed to download dataset data: {res.status_code} {res.text}")
        logger.info(f"📥 Downloading dataset data to {path}")
        url = res.json()["download_uri"]
        path = self.api_client.download_file(url, path)
        logger.info(f"✅ Dataset data downloaded to {path}")
        return path
