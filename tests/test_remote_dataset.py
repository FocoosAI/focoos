from unittest.mock import Mock, mock_open, patch

import pytest

from focoos.ports import DatasetLayout, DatasetPreview, Task
from focoos.remote.remote_dataset import RemoteDataset


@pytest.fixture
def mock_api_client():
    """Fixture to create a mock ApiClient."""
    client = Mock()
    return client


@pytest.fixture
def dataset_preview_data():
    """Fixture to create sample data for DatasetPreview."""
    return {
        "ref": "test-dataset",
        "name": "Test Dataset",
        "layout": DatasetLayout.ROBOFLOW_COCO,
        "task": Task.DETECTION,
        "description": "Test dataset description",
        "spec": {"train_length": 100, "valid_length": 20, "size_mb": 256.0},
    }


@pytest.fixture
def remote_dataset(mock_api_client, dataset_preview_data):
    """Fixture to create a RemoteDataset instance with a mock ApiClient."""
    mock_api_client.get.return_value.json.return_value = dataset_preview_data
    return RemoteDataset("test-dataset", mock_api_client)


def test_init_and_get_info(remote_dataset, mock_api_client, dataset_preview_data):
    """Test initialization and get_info method."""
    assert remote_dataset.ref == "test-dataset"
    assert isinstance(remote_dataset.metadata, DatasetPreview)
    assert remote_dataset.metadata.ref == dataset_preview_data["ref"]
    assert remote_dataset.metadata.name == dataset_preview_data["name"]
    assert remote_dataset.metadata.layout == dataset_preview_data["layout"]
    assert remote_dataset.metadata.task == dataset_preview_data["task"]
    mock_api_client.get.assert_called_once_with("datasets/test-dataset")


def test_delete_success(remote_dataset, mock_api_client):
    """Test successful dataset deletion."""
    mock_api_client.delete.return_value.raise_for_status.return_value = None
    remote_dataset.delete()
    mock_api_client.delete.assert_called_once_with("datasets/test-dataset")


def test_delete_failure(remote_dataset, mock_api_client):
    """Test error handling during deletion."""
    mock_api_client.delete.side_effect = Exception("Delete failed")
    with pytest.raises(Exception, match="Delete failed"):
        remote_dataset.delete()


def test_delete_data_success(remote_dataset, mock_api_client, dataset_preview_data):
    """Test successful data deletion."""
    # Modify data to simulate a dataset without spec after deletion
    updated_preview = dataset_preview_data.copy()
    updated_preview["spec"] = None

    mock_response = Mock()
    mock_response.json.return_value = updated_preview
    mock_api_client.delete.return_value = mock_response

    remote_dataset.delete_data()
    mock_api_client.delete.assert_called_once_with("datasets/test-dataset/data")
    assert remote_dataset.metadata.spec is None


@pytest.mark.parametrize(
    "file_path,expected_error",
    [
        ("nonexistent.zip", FileNotFoundError),
        ("test.txt", ValueError),
    ],
)
def test_upload_data_validation_errors(remote_dataset, file_path, expected_error):
    """Test input validation for data upload."""
    with pytest.raises(expected_error):
        remote_dataset.upload_data(file_path)


@patch("os.path.exists")
@patch("os.path.getsize")
@patch("builtins.open", new_callable=mock_open, read_data="test data")
def test_upload_data_success(
    mock_file, mock_getsize, mock_exists, remote_dataset, mock_api_client, dataset_preview_data
):
    """Test successful data upload."""
    mock_exists.return_value = True
    mock_getsize.return_value = 1024

    # Mock for upload URL generation
    mock_api_client.post.side_effect = [
        Mock(status_code=200, json=lambda: {"url": "https://test-url", "fields": {"key": "value"}}),
        Mock(status_code=422, text="Invalid data"),
    ]
    mock_api_client.external_post.return_value = Mock(status_code=200)

    # Mock for dataset info update
    mock_api_client.get.return_value.json.return_value = dataset_preview_data

    with pytest.raises(ValueError, match="Failed to validate dataset"):
        remote_dataset.upload_data("test.zip")


def test_download_data_success(remote_dataset, mock_api_client):
    """Test successful data download."""
    # Reset previous mock calls
    mock_api_client.get.reset_mock()
    mock_api_client.get.return_value = Mock(status_code=200, json=lambda: {"download_uri": "https://test-download-url"})
    mock_api_client.download_file.return_value = "/local/path/dataset.zip"

    result = remote_dataset.download_data("/local/path")

    assert result == "/local/path/dataset.zip"
    mock_api_client.get.assert_called_once_with("datasets/test-dataset/download")
    mock_api_client.download_file.assert_called_once_with("https://test-download-url", "/local/path")


def test_download_data_failure(remote_dataset, mock_api_client):
    """Test error handling during download."""
    mock_api_client.get.return_value = Mock(status_code=404, text="Not found")

    with pytest.raises(ValueError, match="Failed to download dataset data"):
        remote_dataset.download_data("/local/path")
