import os
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from focoos.config import FOCOOS_CONFIG
from focoos.utils.logger import get_logger
from focoos.utils.system import get_focoos_version

logger = get_logger("HUB")


class ApiClient:
    """
    A simple HTTP client for making GET, POST, and DELETE requests.

    This client is initialized with an API key and a host URL, and it
    automatically includes the API key in the headers of each request.

    Attributes:
        api_key (str): The API key for authorization.
        host_url (str): The base URL for the API.
        default_headers (dict): Default headers including authorization and user agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host_url: Optional[str] = None,
    ):
        """
        Initialize the ApiClient with an API key and host URL.

        Args:
            api_key (str): The API key for authorization.
            host_url (str): The base URL for the API.
        """
        # Use provided api_key if not None, otherwise use config
        self.api_key = api_key if api_key is not None else FOCOOS_CONFIG.focoos_api_key
        self.host_url = host_url or FOCOOS_CONFIG.default_host_url

        self.default_headers = {
            "X-API-Key": self.api_key,
            "user_agent": f"focoos/{get_focoos_version()}",
        }

    def _check_api_key(self):
        if not self.api_key or (isinstance(self.api_key, str) and self.api_key.strip() == ""):
            raise ValueError("API key is required")

    def external_get(self, path: str, params: Optional[dict] = None, stream: bool = False):
        """
        Perform a GET request to an external URL.

        Args:
            path (str): The URL path to request.
            params (Optional[dict], optional): Query parameters for the request. Defaults to None.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Response: The response object from the requests library.
        """
        if params is None:
            params = {}
        return requests.get(path, params=params, stream=stream)

    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        stream: bool = False,
    ):
        """
        Perform a GET request to the specified path on the host URL.

        Args:
            path (str): The URL path to request.
            params (Optional[dict], optional): Query parameters for the request. Defaults to None.
            extra_headers (Optional[dict], optional): Additional headers to include in the request. Defaults to None.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Response: The response object from the requests library.
        """
        self._check_api_key()
        url = f"{self.host_url}/{path}"
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return requests.get(url, headers=headers, params=params, stream=stream)

    def post(
        self,
        path: str,
        data: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        files=None,
    ):
        """
        Perform a POST request to the specified path on the host URL.

        Args:
            path (str): The URL path to request.
            data (Optional[dict], optional): The JSON data to send in the request body. Defaults to None.
            extra_headers (Optional[dict], optional): Additional headers to include in the request. Defaults to None.
            files (optional): Files to send in the request. Defaults to None.

        Returns:
            Response: The response object from the requests library.
        """
        self._check_api_key()
        url = f"{self.host_url}/{path}"
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(url, headers=headers, json=data, files=files)

    def patch(
        self,
        path: str,
        data: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ):
        self._check_api_key()
        url = f"{self.host_url}/{path}"
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return requests.patch(url, headers=headers, json=data)

    def external_post(
        self,
        path: str,
        data: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        files=None,
        stream: bool = False,
    ):
        """
        Perform a POST request to an external URL without using the default host URL.

        This method is used for making POST requests to external services, such as file upload
        endpoints that require direct access without the base host URL prefix.

        Args:
            path (str): The complete URL to send the request to.
            data (Optional[dict], optional): The JSON data to send in the request body. Defaults to None.
            extra_headers (Optional[dict], optional): Headers to include in the request. Defaults to None.
            files (optional): Files to send in the request. Defaults to None.

        Returns:
            Response: The response object from the requests library.
        """
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(path, headers=headers, json=data, files=files, stream=stream)

    def delete(self, path: str, extra_headers: Optional[dict] = None):
        """
        Perform a DELETE request to the specified path on the host URL.

        Args:
            path (str): The URL path to request.
            extra_headers (Optional[dict], optional): Additional headers to include in the request. Defaults to None.

        Returns:
            Response: The response object from the requests library.
        """
        self._check_api_key()
        url = f"{self.host_url}/{path}"
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return requests.delete(url, headers=headers)

    def upload_file(self, path: str, file_path: str, file_size: int):
        """
        Upload a file to the specified path.

        Args:
            path (str): The API endpoint path to upload to
            file_path (str): Path to the file to upload
            file_size (int): Size of the file in bytes

        Returns:
            Response: The response from the upload request
        """
        self._check_api_key()
        return self.post(path, data={"path": file_path, "file_size_bytes": file_size})

    def download_ext_file(self, uri: str, file_dir: str, file_name: Optional[str] = None, skip_if_exists: bool = False):
        """
        Download a file from a URI to a local directory.

        Args:
            uri (str): The URI to download the file from
            file_dir (str): Local directory to save the downloaded file

        Returns:
            str: Path to the downloaded file

        Raises:
            ValueError: If the download fails or filename cannot be determined
        """
        if os.path.exists(file_dir) and not os.path.isdir(file_dir):
            raise ValueError(f"Path is not a directory: {file_dir}")
        if not os.path.exists(file_dir):
            logger.debug(f"📥 Creating directory: {file_dir}")
            os.makedirs(file_dir)
        parsed_url = urlparse(uri)
        file_name = file_name or os.path.basename(parsed_url.path)
        res = self.external_get(uri, stream=True)
        if res.status_code != 200:
            logger.error(f"Failed to download file {file_name}: {res.status_code} {res.text}")
            raise ValueError(f"Failed to download file {file_name}: {res.status_code} {res.text}")

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = os.path.join(file_dir, file_name)
        if skip_if_exists and os.path.exists(file_path):
            logger.debug(f"📥 File already exists: {file_path}")
            return file_path
        total_size = int(res.headers.get("content-length", 0))
        with (
            open(file_path, "wb") as f,
            tqdm(
                desc=file_name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        logger.debug(f"📥 File downloaded: {file_path} Size: {total_size / (1024**2):.2f} MB")
        return file_path
