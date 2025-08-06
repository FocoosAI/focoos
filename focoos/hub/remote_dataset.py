import os
from typing import Optional

from focoos.ports import DATASETS_DIR, DatasetPreview, DatasetSpec
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
        if res.status_code != 200:
            raise ValueError(f"Failed to get dataset info: {res.status_code} {res.text}")
        return DatasetPreview.from_json(res.json())

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

        # Use context manager to properly handle file closure
        with open(path, "rb") as file_obj:
            fields["file"] = (file_name, file_obj, "application/zip")

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
        if complete_upload.status_code not in [200, 201, 204]:
            raise ValueError(f"Failed to validate dataset: {complete_upload.status_code} {complete_upload.text}")
        self.metadata = self.get_info()
        logger.info(f"✅ Dataset validated! => {self.metadata.spec}")
        return self.metadata.spec

    @property
    def name(self):
        return self.metadata.name

    @property
    def task(self):
        return self.metadata.task

    @property
    def layout(self):
        return self.metadata.layout

    def download_data(self, path: str = DATASETS_DIR):
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
        url = res.json()["download_uri"]

        path = self.api_client.download_ext_file(url, path, skip_if_exists=True)
        logger.info(f"✅ Dataset data downloaded to {path}")
        return path

    def __str__(self):
        return f"RemoteDataset(ref={self.ref}, name={self.name}, task={self.task}, layout={self.layout})"
