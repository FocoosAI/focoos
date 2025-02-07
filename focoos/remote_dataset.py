import os
from typing import Optional

from focoos.ports import DatasetPreview, DatasetSpec
from focoos.utils.api_client import ApiClient
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class RemoteDataset:
    def __init__(self, ref: str, api_client: ApiClient):
        self.ref = ref
        self.api_client = api_client
        self.metadata: DatasetPreview = self.get_info()

    def get_info(self) -> DatasetPreview:
        res = self.api_client.get(f"datasets/{self.ref}")
        return DatasetPreview.from_json(res.json())

    def delete(self):
        try:
            res = self.api_client.delete(f"datasets/{self.ref}")
            res.raise_for_status()
            logger.info("Deleted dataset")
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            raise e

    def delete_data(self):
        try:
            res = self.api_client.delete(f"datasets/{self.ref}/data")

            res.raise_for_status()
            new_metadata = DatasetPreview.from_json(res.json())
            self.metadata = new_metadata
            logger.info("Deleted dataset data")
        except Exception as e:
            logger.error(f"Failed to delete dataset data: {e}")

    def upload_data(self, path: str) -> Optional[DatasetSpec]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if not path.endswith(".zip"):
            raise ValueError("Dataset must be .zip compressed")
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
        logger.info("✅ Upload Done.")
        if res.status_code not in [200, 201, 204]:
            raise ValueError(f"Failed to upload dataset: {res.status_code} {res.text}")

        logger.info("🔗 Validate dataset..")
        complete_upload = self.api_client.post(
            f"datasets/{self.ref}/complete-upload",
        )
        complete_upload.raise_for_status()
        logger.info("✅ Done.")
        self.metadata = self.get_info()
        return self.metadata.spec
