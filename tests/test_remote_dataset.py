from unittest.mock import Mock, mock_open, patch

import pytest

from focoos.hub.api_client import ApiClient
from focoos.hub.remote_dataset import RemoteDataset
from focoos.ports import DatasetLayout, DatasetPreview, Task


@pytest.fixture
def mock_api_client():
    """Fixture to create a mock ApiClient."""
    client = Mock(spec=ApiClient)
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
    # Mock the API response correctly
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = dataset_preview_data
    mock_api_client.get.return_value = mock_response

    return RemoteDataset("test-dataset", mock_api_client)


def test_init_and_get_info(remote_dataset, mock_api_client, dataset_preview_data):
    """Test initialization and get_info method."""
    assert remote_dataset.ref == "test-dataset"
    assert isinstance(remote_dataset.metadata, DatasetPreview)
    assert remote_dataset.metadata.ref == dataset_preview_data["ref"]
    assert remote_dataset.metadata.name == dataset_preview_data["name"]
    assert remote_dataset.metadata.layout == dataset_preview_data["layout"]
    assert remote_dataset.metadata.task == dataset_preview_data["task"]
    mock_api_client.get.assert_called_with("datasets/test-dataset")


def test_get_info_failure(mock_api_client):
    """Test get_info method failure."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Not found"
    mock_api_client.get.return_value = mock_response

    with pytest.raises(ValueError, match="Failed to get dataset info"):
        RemoteDataset("nonexistent-dataset", mock_api_client)


def test_properties(remote_dataset, dataset_preview_data):
    """Test dataset properties."""
    assert remote_dataset.name == dataset_preview_data["name"]
    assert remote_dataset.task == dataset_preview_data["task"]
    assert remote_dataset.layout == dataset_preview_data["layout"]


def test_str_representation(remote_dataset):
    """Test string representation."""
    str_repr = str(remote_dataset)
    assert "RemoteDataset" in str_repr
    assert "test-dataset" in str_repr
    assert "Test Dataset" in str_repr


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
    mock_upload_response = Mock()
    mock_upload_response.status_code = 200
    mock_upload_response.json.return_value = {"url": "https://test-url", "fields": {"key": "value"}}

    # Mock for upload completion
    mock_complete_response = Mock()
    mock_complete_response.status_code = 200

    # Mock for external post (file upload)
    mock_external_post_response = Mock()
    mock_external_post_response.status_code = 200

    # Mock updated dataset info
    updated_data = dataset_preview_data.copy()
    updated_data["spec"] = {"train_length": 150, "valid_length": 30, "size_mb": 512.0}
    mock_info_response = Mock()
    mock_info_response.status_code = 200
    mock_info_response.json.return_value = updated_data

    mock_api_client.post.side_effect = [mock_upload_response, mock_complete_response]
    mock_api_client.external_post.return_value = mock_external_post_response
    mock_api_client.get.return_value = mock_info_response

    result = remote_dataset.upload_data("test.zip")

    assert result is not None
    assert result.train_length == 150
    assert result.valid_length == 30

    # Verify API calls
    assert mock_api_client.post.call_count == 2
    mock_api_client.external_post.assert_called_once()


@patch("os.path.exists")
@patch("os.path.getsize")
def test_upload_data_upload_url_failure(mock_getsize, mock_exists, remote_dataset, mock_api_client):
    """Test upload data when URL generation fails."""
    mock_exists.return_value = True
    mock_getsize.return_value = 1024

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Server error"
    mock_api_client.post.return_value = mock_response

    with pytest.raises(ValueError, match="Failed to generate upload url"):
        remote_dataset.upload_data("test.zip")


@patch("os.path.exists")
@patch("os.path.getsize")
@patch("builtins.open", new_callable=mock_open, read_data="test data")
def test_upload_data_external_upload_failure(mock_file, mock_getsize, mock_exists, remote_dataset, mock_api_client):
    """Test upload data when external upload fails."""
    mock_exists.return_value = True
    mock_getsize.return_value = 1024

    # Mock successful URL generation
    mock_upload_response = Mock()
    mock_upload_response.status_code = 200
    mock_upload_response.json.return_value = {"url": "https://test-url", "fields": {"key": "value"}}

    # Mock failed external upload
    mock_external_response = Mock()
    mock_external_response.status_code = 500
    mock_external_response.text = "Upload failed"

    mock_api_client.post.return_value = mock_upload_response
    mock_api_client.external_post.return_value = mock_external_response

    with pytest.raises(ValueError, match="Failed to upload dataset"):
        remote_dataset.upload_data("test.zip")


@patch("os.path.exists")
@patch("os.path.getsize")
@patch("builtins.open", new_callable=mock_open, read_data="test data")
def test_upload_data_validation_failure(mock_file, mock_getsize, mock_exists, remote_dataset, mock_api_client):
    """Test upload data when validation fails."""
    mock_exists.return_value = True
    mock_getsize.return_value = 1024

    # Mock successful URL generation and upload
    mock_upload_response = Mock()
    mock_upload_response.status_code = 200
    mock_upload_response.json.return_value = {"url": "https://test-url", "fields": {"key": "value"}}

    mock_external_response = Mock()
    mock_external_response.status_code = 200

    # Mock failed validation
    mock_validation_response = Mock()
    mock_validation_response.status_code = 422
    mock_validation_response.text = "Invalid data"

    mock_api_client.post.side_effect = [mock_upload_response, mock_validation_response]
    mock_api_client.external_post.return_value = mock_external_response

    with pytest.raises(ValueError, match="Failed to validate dataset"):
        remote_dataset.upload_data("test.zip")


def test_download_data_success(remote_dataset, mock_api_client):
    """Test successful data download."""
    # Reset previous mock calls
    mock_api_client.reset_mock()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"download_uri": "https://test-download-url"}
    mock_api_client.get.return_value = mock_response

    # Mock download_ext_file method
    mock_api_client.download_ext_file.return_value = "/local/path/dataset.zip"

    result = remote_dataset.download_data("/local/path")

    assert result == "/local/path/dataset.zip"
    mock_api_client.get.assert_called_once_with("datasets/test-dataset/download")
    mock_api_client.download_ext_file.assert_called_once_with(
        "https://test-download-url", "/local/path", skip_if_exists=True
    )


def test_download_data_failure(remote_dataset, mock_api_client):
    """Test error handling during download."""
    mock_api_client.reset_mock()

    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Not found"
    mock_api_client.get.return_value = mock_response

    with pytest.raises(ValueError, match="Failed to download dataset data"):
        remote_dataset.download_data("/local/path")


def test_download_data_default_path(remote_dataset, mock_api_client):
    """Test download with default path."""
    mock_api_client.reset_mock()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"download_uri": "https://test-download-url"}
    mock_api_client.get.return_value = mock_response

    mock_api_client.download_ext_file.return_value = "/default/path/dataset.zip"

    result = remote_dataset.download_data()

    assert result == "/default/path/dataset.zip"
    # Should use the default DATASETS_DIR
    mock_api_client.download_ext_file.assert_called_once()
