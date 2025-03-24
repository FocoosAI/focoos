import importlib.metadata as metadata
import os
import platform
import subprocess
from typing import List, Optional

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
    return None


def get_gpu_name() -> Optional[str]:
    """
    Retrieve the name of the first available GPU using nvidia-smi.

    This function runs the `nvidia-smi` command to fetch GPU information.
    If the command executes successfully and a GPU is found, it returns the
    name of the first GPU as a string. If the command fails or no GPU is found,
    it returns None.

    Returns:
        Optional[str]: The name of the first GPU if available, otherwise None.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split("\n")
            if gpus:
                return gpus[0].strip()
    except FileNotFoundError as err:
        logger.warning("nvidia-smi command not found: %s", err)
    return None


def get_gpu_driver() -> Optional[str]:
    """
    Retrieve the GPU driver version using nvidia-smi.

    This function runs the `nvidia-smi` command to fetch driver information.
    If the command executes successfully and a driver is found, it returns the
    driver version as a string. If the command fails or no driver is found,
    it returns None.

    Returns:
        Optional[str]: The GPU driver version if available, otherwise None.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except FileNotFoundError as err:
        logger.warning("nvidia-smi command not found: %s", err)
    return None


def get_gpu_info() -> List[GPUInfo]:
    """
    Retrieve detailed information about all available GPUs using nvidia-smi.

    This function runs multiple `nvidia-smi` commands to fetch GPU information including
    ID, name, memory usage, temperature, and utilization.

    Returns:
        List[GPUInfo]: A list of GPUInfo objects containing detailed information for each GPU.
                      Returns an empty list if no GPUs are available or if an error occurs.
    """
    gpus_info = []
    try:
        # Get GPU names
        name_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Get memory info
        memory_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Get temperature and utilization
        temp_util_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,utilization.gpu", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if name_result.returncode == 0 and memory_result.returncode == 0 and temp_util_result.returncode == 0:
            # Parse GPU names
            gpu_names = {}
            for line in name_result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        idx, name = int(parts[0].strip()), parts[1].strip()
                        gpu_names[idx] = name

            # Parse memory info
            gpu_memory = {}
            for line in memory_result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) == 3:
                        try:
                            idx = int(parts[0].strip())
                            # Check for [N/A] values and set to None
                            if "[N/A]" in parts[1] or "[N/A]" in parts[2]:
                                logger.warning(f"Memory information not available for GPU {idx}")
                                gpu_memory[idx] = (None, None)
                                continue
                            # Convert MiB to GB
                            total = float(parts[1].strip().split()[0]) / 1024
                            used = float(parts[2].strip().split()[0]) / 1024
                            gpu_memory[idx] = (total, used)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse memory info: {line.strip()} - {e}")
                            gpu_memory[idx] = (None, None)

            # Parse temperature and utilization
            gpu_temp_util = {}
            for line in temp_util_result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) == 3:
                        try:
                            idx = int(parts[0].strip())
                            # Check for [N/A] values and set to None
                            if "[N/A]" in parts[1] or "[N/A]" in parts[2]:
                                logger.warning(f"Temperature/utilization not available for GPU {idx}")
                                gpu_temp_util[idx] = (None, None)
                                continue
                            temp = float(parts[1].strip())
                            # Strip % if present
                            util = float(parts[2].strip().rstrip("%"))
                            gpu_temp_util[idx] = (temp, util)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse temp/util info: {line.strip()} - {e}")
                            gpu_temp_util[idx] = (None, None)

            # Create GPUInfo objects
            for idx in gpu_names:
                # Always include GPU if we have its name
                # Add defaults for missing data
                if idx not in gpu_memory:
                    gpu_memory[idx] = (None, None)
                if idx not in gpu_temp_util:
                    gpu_temp_util[idx] = (None, None)

                total_mem, used_mem = gpu_memory[idx]
                temp, util = gpu_temp_util[idx]

                # Calculate used percentage only if values are available
                memory_used_percentage = None
                if total_mem is not None and used_mem is not None and total_mem > 0:
                    memory_used_percentage = round(used_mem / total_mem * 100, 3)

                gpus_info.append(
                    GPUInfo(
                        gpu_id=idx,
                        gpu_name=gpu_names[idx],
                        gpu_memory_total_gb=round(total_mem, 3) if total_mem is not None else None,
                        gpu_memory_used_percentage=memory_used_percentage,
                        gpu_temperature=temp,
                        gpu_load_percentage=util,
                    )
                )

    except FileNotFoundError as err:
        logger.warning("nvidia-smi command not found: %s", err)
    except (ValueError, IndexError, KeyError) as err:
        logger.warning("Error parsing nvidia-smi output: %s", err)

    return gpus_info


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
    gpus_info = get_gpu_info()
    gpu_count = len(gpus_info)
    gpu_driver = get_gpu_driver() if gpu_count > 0 else None

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
