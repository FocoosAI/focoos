import importlib.metadata as metadata
import os
import platform
import subprocess
from typing import Optional

import GPUtil

try:
    import onnxruntime as ort
except ImportError:
    ort = None
import psutil

from focoos.config import FOCOOS_CONFIG
from focoos.ports import GPUInfo, SystemInfo
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


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
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            output = result.stdout
            for line in output.splitlines():
                if "CUDA Version" in line:
                    cuda_version = line.split(":")[-1].strip()
                    cuda_version = cuda_version.split()[0]
                    return cuda_version
    except FileNotFoundError as err:
        logger.warning("nvidia-smi command not found: %s", err)


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
    Collect and return detailed system information.

    This function gathers a wide range of system metrics, including CPU, memory,
    disk, and GPU details, as well as versions of installed packages. The collected
    information is encapsulated in a SystemInfo object.

    Returns:
        SystemInfo: An object containing comprehensive details about the system's
        hardware and software configuration, such as:
            - System and node name
            - CPU type and number of cores
            - Available ONNXRuntime providers
            - Memory and disk usage statistics
            - Number of GPUs, driver, and CUDA version
            - Detailed information for each GPU, if available
            - Versions of key installed packages
            - Environment variables related to the system
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
        "torch",
        "torchvision",
        "nvidia-cuda-runtime-cu12",
        "tensorrt",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"

    environments_var = [
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "FOCOOS_LOG_LEVEL",
        "DEFAULT_HOST_URL",
    ]
    environments = {}
    for var in environments_var:
        environments[var] = os.getenv(var, "")

    return SystemInfo(
        focoos_host=FOCOOS_CONFIG.default_host_url,
        system=system_info.system,
        system_name=system_info.node,
        cpu_type=system_info.machine,
        cpu_cores=psutil.cpu_count(logical=True),
        available_providers=ort.get_available_providers() if ort else None,
        memory_gb=round(memory_info.total / (1024**3), 3),
        memory_used_percentage=round(memory_info.percent, 3),
        disk_space_total_gb=round(disk_info.total / (1024**3), 3),
        disk_space_used_percentage=round(disk_info.percent, 3),
        gpu_count=gpu_count,
        gpu_driver=gpu_driver,
        gpu_cuda_version=get_cuda_version(),
        gpus_info=gpus_info,
        packages_versions=versions,
        environment=environments,
    )
