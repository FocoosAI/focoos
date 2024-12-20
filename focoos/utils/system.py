import importlib.metadata as metadata
import platform
import subprocess
from typing import Optional

import GPUtil
import onnxruntime as ort
import psutil
import requests

from focoos.config import FOCOOS_CONFIG
from focoos.ports import GPUInfo, SystemInfo


class HttpClient:
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
        api_key: str,
        host_url: str,
    ):
        """
        Initialize the HttpClient with an API key and host URL.

        Args:
            api_key (str): The API key for authorization.
            host_url (str): The base URL for the API.
        """
        self.api_key = api_key
        self.host_url = host_url

        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "user_agent": "focoos/0.0.1",
        }

    def get_external_url(
        self, path: str, params: Optional[dict] = None, stream: bool = False
    ):
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
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(url, headers=headers, json=data, files=files)

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
        headers = self.default_headers
        if extra_headers:
            headers.update(extra_headers)
        return requests.delete(url, headers=headers)


def get_cuda_version() -> Optional[str]:
    """
    Retrieve the CUDA version installed on the system.

    This function runs the `nvidia-smi` command to fetch the CUDA version.
    If the command executes successfully and the CUDA version is found in the output,
    it returns the version as a string. If the command fails or the CUDA version is not found,
    it returns None.

    Returns:
        Optional[str]: The CUDA version if available, otherwise None.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0:
            output = result.stdout
            for line in output.splitlines():
                if "CUDA Version" in line:
                    cuda_version = line.split(":")[-1].strip()
                    cuda_version = cuda_version.split()[0]
                    return cuda_version
            return None
        else:
            return None
    except FileNotFoundError:
        return None


def get_gpu_name() -> Optional[str]:
    """
    Retrieve the name of the first available GPU.

    This function uses the GPUtil library to get the name of the first GPU detected.
    If no GPUs are available, it returns None.

    Returns:
        Optional[str]: The name of the first GPU if available, otherwise None.
    """
    try:
        return GPUtil.getGPUs()[0].name
    except IndexError:
        return None


def get_cpu_name() -> Optional[str]:
    """
    Retrieve the name of the CPU.

    This function uses the psutil library to get the name of the CPU.
    If no CPU is available, it returns None.

    Returns:
        Optional[str]: The name of the CPU if available, otherwise None.
    """
    return platform.processor()


def get_system_info() -> SystemInfo:
    """
    Gather and return comprehensive system information.

    This function collects various system metrics including CPU, memory, disk,
    and GPU details, as well as installed package versions. It returns this
    information encapsulated in a SystemInfo object.

    Returns:
        SystemInfo: An object containing detailed information about the system's
        hardware and software configuration, including:
            - System and node name
            - CPU type and core count
            - Available ONNXRuntime providers
            - Memory and disk usage statistics
            - GPU count, driver, and CUDA version
            - Detailed GPU information if available
            - Versions of key installed packages
    """
    system_info = platform.uname()
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("/")
    gpu_info = GPUtil.getGPUs()
    if len(gpu_info) == 0:
        gpu_count = 0
        gpu_driver = None
        gpus_info = None
    else:
        gpu_count = len(gpu_info)
        gpu_driver = gpu_info[0].driver
        gpus_info = []
        for i, gpu in enumerate(gpu_info):
            gpus_info.append(
                GPUInfo(
                    gpu_id=i,
                    gpu_name=gpu.name,
                    gpu_memory_total_gb=round(gpu.memoryTotal / 1024, 3),
                    gpu_memory_used_percentage=round(gpu.memoryUsed / 1024, 3),
                    gpu_temperature=gpu.temperature,
                    gpu_load_percentage=gpu.load * 100,
                )
            )
    packages = [
        "focoos",
        "tensorrt",
        "onnxruntime",
        "onnxruntime-gpu",
        "numpy",
        "opencv-python",
        "pillow",
        "supervision",
        "pydantic",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"

    return SystemInfo(
        focoos_host=FOCOOS_CONFIG.default_host_url,
        system=system_info.system,
        system_name=system_info.node,
        cpu_type=system_info.machine,
        cpu_cores=psutil.cpu_count(logical=True),
        available_providers=ort.get_available_providers(),
        memory_gb=round(memory_info.total / (1024**3), 3),
        memory_used_percentage=round(memory_info.percent, 3),
        disk_space_total_gb=round(disk_info.total / (1024**3), 3),
        disk_space_used_percentage=round(disk_info.percent, 3),
        gpu_count=gpu_count,
        gpu_driver=gpu_driver,
        gpu_cuda_version=get_cuda_version(),
        gpus_info=gpus_info,
        packages_versions=versions,
    )
