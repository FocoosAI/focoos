from unittest.mock import MagicMock, patch

import pytest

from focoos.hub.api_client import ApiClient
from focoos.ports import GPUDevice, GPUInfo, SystemInfo
from focoos.utils.system import (
    get_cpu_name,
    get_cuda_version,
    get_system_info,
)


@pytest.fixture
def extra_headers():
    return {"X-Dummy-Header": "DummyValue"}


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


def test_get_cpu_name():
    with patch("platform.processor", return_value="Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"):
        assert get_cpu_name() == "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"


def test_get_system_info():
    with (
        patch("focoos.utils.system.get_gpu_info") as mock_get_gpu_info,
        patch("focoos.utils.system.get_cuda_version") as mock_get_cuda_version,
    ):
        # Mock the GPU-related functions to avoid real nvidia-smi calls
        mock_get_gpu_info.return_value = GPUInfo(
            gpu_count=0,
            gpu_driver="533.104.00",
            gpu_cuda_version="12.1",
            devices=[
                GPUDevice(
                    gpu_id=0,
                    gpu_name="NVIDIA RTX 4090",
                    gpu_memory_total_gb=24.0,
                    gpu_memory_used_percentage=70.0,
                    gpu_temperature=65.0,
                    gpu_load_percentage=80.0,
                )
            ],
        )
        mock_get_cuda_version.return_value = None

        system_info = get_system_info()
        assert isinstance(system_info, SystemInfo)
        assert system_info.system is not None
        assert system_info.cpu_cores is not None
        assert system_info.cpu_cores > 0
        assert system_info.gpu_info is not None
        assert system_info.gpu_info.gpu_count == 0


def test_api_client_get_external_url():
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        response = client.external_get("test/path")
        assert response.status_code == 200
        mock_get.assert_called_with("test/path", params={}, stream=False)


def test_api_client_get(extra_headers):
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        response = client.get("test/path", extra_headers=extra_headers)
        assert response.status_code == 200
        mock_get.assert_called_with(
            "http://example.com/test/path",
            headers={**client.default_headers, **extra_headers},
            params=None,
            stream=False,
        )


def test_api_client_post(extra_headers):
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 201
        response = client.post("test/path", data={"key": "value"}, extra_headers=extra_headers)
        assert response.status_code == 201
        mock_post.assert_called_with(
            "http://example.com/test/path",
            headers={**client.default_headers, **extra_headers},
            json={"key": "value"},
            files=None,
        )


def test_api_client_delete(extra_headers):
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value.status_code = 204
        response = client.delete("test/path", extra_headers=extra_headers)
        assert response.status_code == 204
        mock_delete.assert_called_with(
            "http://example.com/test/path",
            headers={**client.default_headers, **extra_headers},
        )
