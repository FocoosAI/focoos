import importlib.metadata as metadata
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch

from focoos.ports import GPUInfo
from focoos.utils.distributed import comm

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
    except FileNotFoundError:
        logger.warning("nvidia-smi not available")
        return None


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
            gpu_info.total_gpu_memory_gb = sum(device.gpu_memory_total_gb for device in gpus_device)
    except FileNotFoundError as err:
        logger.warning("nvidia-smi command not found: %s", err)
    except Exception as err:
        logger.warning("Error parsing nvidia-smi output: %s", err)

    return gpu_info


def get_cpu_name() -> str:
    """
    Retrieve the name of the CPU.

    This function uses the psutil library to get the name of the CPU.
    If no CPU is available, it returns None.

    Returns:
        Optional[str]: The name of the CPU if available, otherwise None.
    """
    return platform.processor()


def get_focoos_version() -> str:
    return metadata.version("focoos")


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
        "fvcore",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    focoos_version = get_focoos_version()
    environments_var = [
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "FOCOOS_LOG_LEVEL",
    ]
    environments = {}
    for var in environments_var:
        environments[var] = os.getenv(var, "")

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        torch_cuda_home = CUDA_HOME
        torch_cudnn_version = torch.backends.cudnn.version()
        torch_info = f"{torch.__version__} cudnn: {torch_cudnn_version} cuda home: {torch_cuda_home} root: {os.path.dirname(torch.__file__)}"
    except Exception as e:
        logger.warning(f"Error getting torch cuda home: {e}")
        torch_info = None

    ort_providers = ort.get_available_providers() if ort else None
    return SystemInfo(
        focoos_host=FOCOOS_CONFIG.default_host_url,
        focoos_version=focoos_version,
        python_version=sys.version.replace("\n", ""),
        system=system_info.system,
        system_name=system_info.node,
        pytorch_info=torch_info,
        cpu_type=system_info.machine,
        cpu_cores=psutil.cpu_count(logical=True),
        available_onnx_providers=ort_providers,
        memory_gb=round(memory_info.total / (1024**3), 3),
        memory_used_percentage=round(memory_info.percent, 3),
        disk_space_total_gb=round(disk_info.total / (1024**3), 3),
        disk_space_used_percentage=round(disk_info.percent, 3),
        gpu_info=gpu_info,
        packages_versions=versions,
        environment=environments,
    )


def check_folder_exists(folder_path: Union[str, Path]) -> bool:
    """
    Check if a specified folder exists.

    Parameters:
        folder_path (Union[str, Path]): The path to the folder to check.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    folder_path = Path(folder_path)
    return folder_path.is_dir()


def is_inside_sagemaker():
    res = os.environ.get("SM_HOSTS") is not None
    return res


def list_directories(base_directory: Union[str, Path]) -> List[Path]:
    """
    A function that lists directories within a base directory.

    Parameters:
    - base_directory: A Union of str or Path, the base directory to list directories from.

    Returns:
    - List[Path]: A list of Path objects representing directories within the base directory.
    """
    base_directory = Path(base_directory)
    directories = [child for child in base_directory.iterdir() if child.is_dir()]
    return directories


def extract_archive(
    archive_path: str, destination: Optional[str] = None, delete_original: bool = False
) -> Union[str, Path]:
    """
    Extract an archive to a specified destination or the same folder.

    This function supports extracting .zip, .tar.gz, and .tar files.

    Args:
        archive_path (str): The path to the archive file to be extracted.
        destination (Optional[str]): The path where the archive should be extracted.
            If None, the archive will be extracted to its current directory.
            Defaults to None.
        delete_original (bool): If True, deletes the original archive file after extraction.
            Defaults to False.

    Returns:
        str: The path to the directory where the archive was extracted.

    Raises:
        ValueError: If the archive format is not supported.

    Note:
        The function logs the start and end of the extraction process, including the time taken.
    """

    # Determine the extraction path
    t0 = time.time()
    base_dir = os.path.dirname(archive_path)
    extracted_dir = base_dir
    if destination is not None:
        extracted_dir = os.path.join(base_dir, destination)

    if comm.is_main_process():
        logger.info(f"Extracting archive: {archive_path} to {extracted_dir}")

        # Create the extracted directory
        os.makedirs(extracted_dir, exist_ok=True)

        # Get the file extension
        file_extension = get_file_extension(archive_path)

        # Extract the archive
        if file_extension == "application/zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extracted_dir)
        elif file_extension == "application/gzip":
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(extracted_dir)
        elif file_extension == "application/x-tar":
            with tarfile.open(archive_path, "r:") as tar_ref:
                tar_ref.extractall(extracted_dir)
        else:
            raise ValueError("Unsupported archive format. Only .zip and .tar.gz are supported.")
        t1 = time.time()
        logger.info(f"[elapsed {t1 - t0:.3f} ] Extracted archive to: {extracted_dir}")

    comm.synchronize()
    # Remove __MACOSX directory
    if "__MACOSX" in os.listdir(extracted_dir):
        shutil.rmtree(os.path.join(extracted_dir, "__MACOSX"))

    if len(list_directories(extracted_dir)) == 1:
        extracted_dir = list_directories(extracted_dir)[0]

    POSSIBLE_TRAIN_DIRS = ["train", "training"]
    POSSIBLE_VAL_DIRS = ["valid", "val", "validation"]
    inner_dirs = list_directories(extracted_dir)
    if not any(dir.name in POSSIBLE_TRAIN_DIRS for dir in inner_dirs):
        raise FileNotFoundError(
            f"Train split not found in {extracted_dir}: {[str(x) for x in inner_dirs]}. You should provide a zip dataset with only a root folder or train and val subfolders."
        )
    if not any(dir.name in POSSIBLE_VAL_DIRS for dir in inner_dirs):
        raise FileNotFoundError(
            f"Validation split not found in {extracted_dir}: {[str(x) for x in inner_dirs]}. You should provide a zip dataset with only a root folder or train and val subfolders."
        )

    # Optionally delete the original archive
    if delete_original:
        os.remove(archive_path)

    return extracted_dir


def get_file_extension(file_path):
    """
    Determine the MIME type of a file based on its extension.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MIME type of the file
    """
    extension = os.path.splitext(file_path)[1].lower()

    # Map common extensions to MIME types
    mime_types = {
        ".zip": "application/zip",
        ".gz": "application/gzip",
        ".tar": "application/x-tar",
        ".tar.gz": "application/gzip",
        ".tgz": "application/gzip",
    }

    # Check for .tar.gz extension first
    if file_path.lower().endswith(".tar.gz") or file_path.lower().endswith(".tgz"):
        mime_type = "application/gzip"
    else:
        mime_type = mime_types.get(extension, "application/octet-stream")

    logger.debug(f"Supposed file extension: {mime_type}")
    return mime_type


def list_files_with_extensions(base_dir: Union[str, Path], extensions: Optional[List[str]] = None) -> List[Path]:
    """
    A function that lists files in a directory based on the provided extensions.

    Parameters:
    - base_dir: Union[str, Path] - The base directory where the files will be listed.
    - extensions: Optional[List[str]] - A list of file extensions to filter the files by.

    Returns:
    - List[Path]: A list of Path objects representing the files in the directory matching the provided extensions.
    """
    base_dir = Path(base_dir)
    if extensions:
        files = []
        for ext in extensions:
            if ext.startswith("."):
                ext = ext[1:]
            _glob = f"*.{ext}"
            files.extend(base_dir.glob(_glob))
    else:
        files = base_dir.glob("*")
    return [file for file in files if file.is_file()]


def get_device_name() -> str:
    gpu_info = get_gpu_info()
    if gpu_info.devices is not None and len(gpu_info.devices) > 0:
        if gpu_info.devices[0].gpu_name is not None:
            return gpu_info.devices[0].gpu_name
        else:
            return "Unknown GPU"
    else:
        cpu_name = get_cpu_name()
        return cpu_name if cpu_name is not None else "CPU"


def get_device_type() -> Literal["cuda", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
