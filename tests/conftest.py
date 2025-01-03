from unittest.mock import patch

import pytest


@pytest.fixture
def mock_http_client():
    """Fixture to create a mock HttpClient."""
    with patch("focoos.focoos.HttpClient") as MockHttpClient:
        mock_client = MockHttpClient.return_value
        yield mock_client
