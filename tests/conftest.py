import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mock_http_client():
    """Fixture to create a mock HttpClient."""
    with patch("focoos.focoos.HttpClient") as MockHttpClient:
        mock_client = MockHttpClient.return_value
        yield mock_client


@pytest.fixture
def pil_image() -> Image.Image:
    """Create a simple RGB image using PIL."""
    image = Image.new("RGB", (100, 100), color="red")
    return image


@pytest.fixture
def image_bytes(pil_image: Image.Image) -> bytes:
    """Convert PIL image to bytes."""
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


@pytest.fixture
def image_path(tmp_path: Path, pil_image: Image.Image) -> Path:
    """Save PIL image to a temporary file and return the file path."""
    img_path = tmp_path / "test_image.png"
    pil_image.save(img_path)
    return img_path


@pytest.fixture
def image_ndarray(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to NumPy array (which is in RGB format)."""
    return np.array(pil_image)
