import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from focoos.utils.api_client import ApiClient


@pytest.fixture
def extra_headers():
    """Fixture to provide extra headers for testing."""
    return {"Custom-Header": "custom-value"}


@pytest.fixture
def api_client():
    """Fixture to provide an ApiClient instance for testing."""
    return ApiClient(api_key="test_key", host_url="http://example.com")


def test_api_client_initialization():
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    assert client.api_key == "test_key"
    assert client.host_url == "http://example.com"
    assert "X-API-Key" in client.default_headers
    assert client.default_headers["X-API-Key"] == "test_key"


def test_api_client_initialization_with_defaults():
    with patch("focoos.utils.api_client.FOCOOS_CONFIG") as mock_config:
        mock_config.focoos_api_key = "default_key"
        mock_config.default_host_url = "http://default.com"
        client = ApiClient()
        assert client.api_key == "default_key"
        assert client.host_url == "http://default.com"


def test_api_client_check_api_key_missing():
    client = ApiClient(api_key="", host_url="http://example.com")
    with pytest.raises(ValueError, match="API key is required"):
        client.get("test/path")


def test_api_client_check_api_key_whitespace():
    client = ApiClient(api_key="   ", host_url="http://example.com")
    with pytest.raises(ValueError, match="API key is required"):
        client.get("test/path")


def test_api_client_check_api_key_none():
    with patch("focoos.utils.api_client.FOCOOS_CONFIG") as mock_config:
        mock_config.focoos_api_key = None
        mock_config.default_host_url = "http://example.com"
        client = ApiClient(api_key=None, host_url="http://example.com")
        with pytest.raises(ValueError, match="API key is required"):
            client.get("test/path")


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


def test_api_client_get_with_params(api_client):
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        params = {"param1": "value1", "param2": "value2"}
        response = api_client.get("test/path", params=params)
        assert response.status_code == 200
        mock_get.assert_called_with(
            "http://example.com/test/path",
            headers=api_client.default_headers,
            params=params,
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


def test_api_client_post_with_files(api_client):
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 201
        files = {"file": ("test.txt", "content")}
        response = api_client.post("test/path", files=files)
        assert response.status_code == 201
        mock_post.assert_called_with(
            "http://example.com/test/path",
            headers=api_client.default_headers,
            json=None,
            files=files,
        )


def test_api_client_patch(api_client):
    with patch("requests.patch") as mock_patch:
        mock_patch.return_value.status_code = 200
        response = api_client.patch("test/path", data={"key": "value"})
        assert response.status_code == 200
        mock_patch.assert_called_with(
            "http://example.com/test/path",
            headers=api_client.default_headers,
            json={"key": "value"},
        )


def test_api_client_patch_with_headers(api_client):
    extra_headers = {"Authorization": "Bearer token"}
    with patch("requests.patch") as mock_patch:
        mock_patch.return_value.status_code = 200
        response = api_client.patch("test/path", data={"key": "value"}, extra_headers=extra_headers)
        assert response.status_code == 200
        mock_patch.assert_called_with(
            "http://example.com/test/path",
            headers={**api_client.default_headers, **extra_headers},
            json={"key": "value"},
        )


def test_api_client_external_post():
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 201
        response = client.external_post("http://external.com/upload", data={"key": "value"})
        assert response.status_code == 201
        mock_post.assert_called_with(
            "http://external.com/upload",
            headers={},
            json={"key": "value"},
            files=None,
            stream=False,
        )


def test_api_client_external_post_with_headers():
    client = ApiClient(api_key="test_key", host_url="http://example.com")
    extra_headers = {"Authorization": "Bearer token"}
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 201
        response = client.external_post("http://external.com/upload", extra_headers=extra_headers)
        assert response.status_code == 201
        mock_post.assert_called_with(
            "http://external.com/upload",
            headers=extra_headers,
            json=None,
            files=None,
            stream=False,
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


def test_api_client_upload_file(api_client):
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        response = api_client.upload_file("upload/file", "/path/to/file.txt", 1024)
        assert response.status_code == 200
        # upload_file calls the post method internally
        mock_post.assert_called_once()


def test_api_client_download_ext_file_success(api_client):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("requests.get") as mock_get:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-length": "1024"}
            mock_response.iter_content.return_value = [b"test content"]
            mock_get.return_value = mock_response

            # Test download
            file_path = api_client.download_ext_file("http://example.com/file.txt", temp_dir, file_name="test_file.txt")

            assert file_path == os.path.join(temp_dir, "test_file.txt")
            assert os.path.exists(file_path)
            with open(file_path, "r") as f:
                content = f.read()
                assert content == "test content"


def test_api_client_download_ext_file_failure(api_client):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("requests.get") as mock_get:
            # Mock failed response
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            mock_get.return_value = mock_response

            # Test download failure
            with pytest.raises(ValueError, match="Failed to download file"):
                api_client.download_ext_file("http://example.com/nonexistent.txt", temp_dir)


def test_api_client_download_ext_file_skip_if_exists(api_client):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create existing file
        existing_file = os.path.join(temp_dir, "existing.txt")
        with open(existing_file, "w") as f:
            f.write("existing content")

        with patch("requests.get") as mock_get:
            # Mock successful response (even though file exists, HTTP request is still made)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-length": "1024"}
            mock_response.iter_content.return_value = [b"new content"]
            mock_get.return_value = mock_response

            # Test skip if exists
            file_path = api_client.download_ext_file(
                "http://example.com/existing.txt", temp_dir, file_name="existing.txt", skip_if_exists=True
            )

            assert file_path == existing_file
            # test that the file is not downloaded again
            mock_get.assert_not_called()
            # Verify original content is preserved
            with open(existing_file, "r") as f:
                content = f.read()
                assert content == "existing content"


def test_api_client_download_ext_file_invalid_directory(api_client):
    with tempfile.NamedTemporaryFile() as temp_file:
        # Try to use a file as directory
        with pytest.raises(ValueError, match="Path is not a directory"):
            api_client.download_ext_file("http://example.com/file.txt", temp_file.name)


def test_api_client_download_ext_file_creates_directory(api_client):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create path to non-existing subdirectory
        non_existing_dir = os.path.join(temp_dir, "new_subdir", "nested_dir")

        with patch("requests.get") as mock_get:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-length": "1024"}
            mock_response.iter_content.return_value = [b"test content"]
            mock_get.return_value = mock_response

            # Test download to non-existing directory
            file_path = api_client.download_ext_file(
                "http://example.com/file.txt", non_existing_dir, file_name="test_file.txt"
            )

            # Verify directory was created
            assert os.path.exists(non_existing_dir)
            assert os.path.isdir(non_existing_dir)

            # Verify file was downloaded to the created directory
            expected_file_path = os.path.join(non_existing_dir, "test_file.txt")
            assert file_path == expected_file_path
            assert os.path.exists(file_path)

            # Verify file content
            with open(file_path, "r") as f:
                content = f.read()
                assert content == "test content"
