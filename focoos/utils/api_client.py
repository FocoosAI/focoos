from typing import Optional

import requests

from focoos.config import FOCOOS_CONFIG
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


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
        api_key: Optional[str] = FOCOOS_CONFIG.focoos_api_key,
        host_url: Optional[str] = FOCOOS_CONFIG.default_host_url,
    ):
        """
        Initialize the ApiClient with an API key and host URL.

        Args:
            api_key (str): The API key for authorization.
            host_url (str): The base URL for the API.
        """
        if not api_key:
            raise ValueError("API key is required")
        if not host_url:
            raise ValueError("Host URL is required")

        self.api_key = api_key
        self.host_url = host_url

        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "user_agent": "focoos/0.0.1",
        }

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
        url = f"{self.host_url}/{path}"
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(url, headers=headers, json=data, files=files)

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
        url = f"{self.host_url}/{path}"
        headers = self.default_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        return requests.delete(url, headers=headers)

    def upload_file(self, path: str, file_path: str, file_size: int):
        return self.post(path, data={"path": file_path, "file_size_bytes": file_size})
