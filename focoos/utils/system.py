import importlib.metadata as metadata
import os
import platform
import subprocess
from typing import Optional

from focoos.ports import GPUInfo

try:
    import onnxruntime as ort
except ImportError:
    ort = None
import psutil

from focoos.config import FOCOOS_CONFIG
from focoos.ports import GPUDevice, SystemInfo
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


def get_gpu_info() -> GPUInfo:
    """
    Retrieve detailed information about all available GPUs using nvidia-smi.

    This function runs a single `nvidia-smi` command to fetch GPU information including
    ID, name, memory usage, temperature, utilization, driver version and CUDA version.

    Returns:
        GPUInfo: An object containing comprehensive GPU information including devices list,
                driver version, CUDA version and GPU count.
    """
    gpu_info = GPUInfo()
    gpus_device = []
    try:
        # Get all GPU information in a single query
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            driver_version = None

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split(",")
                if len(parts) == 7:
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip()

                        # Handle memory values
                        total_mem = None
                        used_mem = None
                        memory_used_percentage = None
                        if "[N/A]" not in parts[2] and "[N/A]" not in parts[3]:
                            total_mem = float(parts[2].strip().split()[0]) / 1024  # Convert MiB to GB
                            used_mem = float(parts[3].strip().split()[0]) / 1024
                            if total_mem > 0:
                                memory_used_percentage = round(used_mem / total_mem * 100, 3)

                        # Handle temperature and utilization
                        temp = None if "[N/A]" in parts[4] else float(parts[4].strip())
                        util = None if "[N/A]" in parts[5] else float(parts[5].strip().rstrip("%"))

                        # Store driver and CUDA version from first GPU (they're the same for all GPUs)
                        if driver_version is None:
                            driver_version = parts[6].strip()

                        gpus_device.append(
                            GPUDevice(
                                gpu_id=idx,
                                gpu_name=name,
                                gpu_memory_total_gb=round(total_mem, 3) if total_mem is not None else None,
                                gpu_memory_used_percentage=memory_used_percentage,
                                gpu_temperature=temp,
                                gpu_load_percentage=util,
                            )
                        )
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse GPU info: {line.strip()} - {e}")

            # Set all GPUInfo fields
            gpu_info.devices = gpus_device
            gpu_info.gpu_count = len(gpus_device)
            gpu_info.gpu_driver = driver_version
            gpu_info.gpu_cuda_version = get_cuda_version()

    except FileNotFoundError as err:
        logger.warning("nvidia-smi command not found: %s", err)
    except Exception as err:
        logger.warning("Error parsing nvidia-smi output: %s", err)

    return gpu_info


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

    # Get GPU information using nvidia-smi
    gpu_info = get_gpu_info()

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
        gpu_info=gpu_info,
        packages_versions=versions,
        environment=environments,
    )
