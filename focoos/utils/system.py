from typing import Optional

import requests

from focoos.config import FOCOOS_CONFIG


class HttpClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        host_url: str = FOCOOS_CONFIG.default_host_url,
    ):
        if not api_key and not FOCOOS_CONFIG.focoos_api_key:
            raise ValueError("API key is required")
        self.api_key = api_key or FOCOOS_CONFIG.focoos_api_key
        self.host_url = host_url

        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "user_agent": "focoos/0.0.1",
        }

    def get_external_url(self, path: str, params: dict = None, stream: bool = False):
        return requests.get(path, params=params, stream=stream)

    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        stream: bool = False,
    ):
        url = f"{self.host_url}/{path}"
        headers = self.default_headers
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
        url = f"{self.host_url}/{path}"
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(url, headers=headers, json=data, files=files)

    def delete(self, path: str, extra_headers: Optional[dict] = None):
        url = f"{self.host_url}/{path}"
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.delete(url, headers=headers)
