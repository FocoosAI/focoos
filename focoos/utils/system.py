import requests


class HttpClient:
    def __init__(self, api_key: str, host_url: str = "https://api.dev.focoos.ai/v0"):
        self.api_key = api_key
        self.host_url = host_url
        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "user_agent": "focoos/0.0.1",
        }

    def get(self, path: str, params: dict = None, extra_headers: dict = None):
        url = f"{self.host_url}/{path}"
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.get(url, headers=headers, params=params)

    def post(
        self, path: str, data: dict = None, extra_headers: dict = None, files=None
    ):
        url = f"{self.host_url}/{path}"
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(url, headers=headers, json=data, files=files)

    def delete(self, path: str, extra_headers: dict = None):
        url = f"{self.host_url}/{path}"
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.delete(url, headers=headers)
