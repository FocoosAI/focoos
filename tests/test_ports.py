import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from focoos.ports import (
    DatasetInfo,
    DatasetLayout,
    GPUInfo,
    Hyperparameters,
    ModelFormat,
    RuntimeTypes,
    SystemInfo,
)


def test_validate_wandb_project_valid():
    wandb_project = "randomname"
    params = Hyperparameters(wandb_project=wandb_project)
    assert params.wandb_project == wandb_project


def test_validate_wandb_project_invalid():
    # Invalid wandb_project values
    invalid_values = [
        "ORG ID/PROJECT NAME",  # Spaces are not allowed
        "ORG@ID/PROJECT#NAME",  # Special characters are not allowed
        "ORG/PROJECT:NAME",  # Special characters are not allowed
    ]
    for value in invalid_values:
        with pytest.raises(ValidationError) as exc_info:
            Hyperparameters(wandb_project=value)
        assert "Wandb project name must only contain characters, dashes, underscores, and dots." in str(exc_info.value)


def test_validate_s3_url_valid():
    url = "s3://bucketname/path"
    dataset = DatasetInfo(url=url, name="test", layout=DatasetLayout.CATALOG)
    assert dataset.url == url


def test_validate_s3_url_invalid():
    url = "invalid_url"
    with pytest.raises(ValidationError) as exc_info:
        DatasetInfo(url=url, name="test", layout=DatasetLayout.CATALOG)
    assert "Invalid S3 URL format, must be s3://BUCKET_NAME/path" in str(exc_info.value)


def test_pretty_print_with_gpus_info(mocker: MockerFixture):
    gpu_info = GPUInfo(
        gpu_id=0,
        gpu_name="NVIDIA GTX 1080",
        gpu_memory_total_gb=8.0,
        gpu_memory_used_percentage=70.0,
        gpu_temperature=65.0,
        gpu_load_percentage=80.0,
    )

    system_info = SystemInfo(
        focoos_host="localhost",
        system="Linux",
        system_name="TestSystem",
        cpu_type="Intel",
        cpu_cores=8,
        memory_gb=16.0,
        memory_used_percentage=50.0,
        available_providers=["provider1"],
        disk_space_total_gb=500.0,
        disk_space_used_percentage=60.0,
        gpu_count=1,
        gpu_driver="NVIDIA",
        gpu_cuda_version="11.2",
        packages_versions={"pytest": "6.2.4", "pydantic": "1.8.2"},
        gpus_info=[gpu_info],
        environment={"FOCOOS_LOG_LEVEL": "DEBUG", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"},
    )

    mock_print = mocker.patch("builtins.print")

    expected_calls = [
        "================ SYSTEM INFO ====================",
        "focoos_host: localhost",
        "system: Linux",
        "system_name: TestSystem",
        "cpu_type: Intel",
        "cpu_cores: 8",
        "memory_gb: 16.0",
        "memory_used_percentage: 50.0",
        "available_providers:",
        "  - provider1",
        "disk_space_total_gb: 500.0",
        "disk_space_used_percentage: 60.0",
        "gpu_count: 1",
        "gpu_driver: NVIDIA",
        "gpu_cuda_version: 11.2",
        "packages_versions:",
        "  - pytest: 6.2.4",
        "  - pydantic: 1.8.2",
        "gpus_info:",
        "- id: 0",
        "    - gpu-name: NVIDIA GTX 1080",
        "    - gpu-memory-total-gb: 8.0",
        "    - gpu-memory-used-percentage: 70.0",
        "    - gpu-temperature: 65.0",
        "    - gpu-load-percentage: 80.0",
        "environment:",
        "  - LD_LIBRARY_PATH: /usr/local/cuda/lib64",
        "  - FOCOOS_LOG_LEVEL: DEBUG",
        "================================================",
    ]

    system_info.pretty_print()

    # Validate that all expected calls were made
    for call in expected_calls:
        mock_print.assert_any_call(call)


@pytest.mark.parametrize(
    "runtime_type,expected_format",
    [
        (RuntimeTypes.ONNX_CUDA32, ModelFormat.ONNX),
        (RuntimeTypes.ONNX_TRT32, ModelFormat.ONNX),
        (RuntimeTypes.ONNX_TRT16, ModelFormat.ONNX),
        (RuntimeTypes.ONNX_CPU, ModelFormat.ONNX),
        (RuntimeTypes.ONNX_COREML, ModelFormat.ONNX),
        (RuntimeTypes.TORCHSCRIPT_32, ModelFormat.TORCHSCRIPT),
    ],
)
def test_model_format_from_runtime_type(runtime_type, expected_format):
    """Test that from_runtime_type returns correct ModelFormat for each RuntimeType"""
    assert ModelFormat.from_runtime_type(runtime_type) == expected_format


def test_model_format_from_runtime_type_invalid():
    """Test that from_runtime_type raises ValueError for invalid runtime type"""
    with pytest.raises(ValueError, match="Invalid runtime type:.*"):
        ModelFormat.from_runtime_type("invalid_runtime")
