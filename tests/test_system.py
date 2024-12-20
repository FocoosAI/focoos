from unittest.mock import MagicMock, patch

import pytest

from focoos.ports import SystemInfo
from focoos.utils.system import (
    HttpClient,
    get_cpu_name,
    get_cuda_version,
    get_gpu_name,
    get_system_info,
)


def test_get_cuda_version():
    with patch("subprocess.run") as mock_run:
        # Simulate successful nvidia-smi command
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2",
        )
        assert get_cuda_version() == "11.2"

        # Simulate nvidia-smi command not found
        mock_run.side_effect = FileNotFoundError
        assert get_cuda_version() is None


def test_get_gpu_name():
    with patch("GPUtil.getGPUs") as mock_get_gpus:
        # Simulate GPU available
        mock_gpu = MagicMock()
        mock_gpu.name = "Tesla T4"
        mock_get_gpus.return_value = [mock_gpu]
        assert get_gpu_name() == "Tesla T4"

        # Simulate no GPU available
        mock_get_gpus.return_value = []
        assert get_gpu_name() is None


def test_get_cpu_name():
    with patch(
        "platform.processor", return_value="Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"
    ):
        assert get_cpu_name() == "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"


def test_get_system_info():
    system_info = get_system_info()
    assert isinstance(system_info, SystemInfo)
    assert system_info.system is not None
    assert system_info.cpu_cores is not None
    assert system_info.cpu_cores > 0


def test_http_client_get_external_url():
    client = HttpClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        response = client.get_external_url("test/path")
        assert response.status_code == 200
        mock_get.assert_called_with("test/path", params={}, stream=False)


def test_http_client_get():
    client = HttpClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        response = client.get("test/path")
        assert response.status_code == 200
        mock_get.assert_called_with(
            "http://example.com/test/path",
            headers=client.default_headers,
            params=None,
            stream=False,
        )


def test_http_client_post():
    client = HttpClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 201
        response = client.post("test/path", data={"key": "value"})
        assert response.status_code == 201
        mock_post.assert_called_with(
            "http://example.com/test/path",
            headers=client.default_headers,
            json={"key": "value"},
            files=None,
        )


def test_http_client_delete():
    client = HttpClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value.status_code = 204
        response = client.delete("test/path")
        assert response.status_code == 204
        mock_delete.assert_called_with(
            "http://example.com/test/path", headers=client.default_headers
        )
