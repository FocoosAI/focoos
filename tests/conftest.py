import datetime
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from focoos.ports import FocoosTask, ModelMetadata, ModelStatus


@pytest.fixture
def mock_api_client():
    """Fixture to create a mock ApiClient."""
    with patch("focoos.focoos.ApiClient") as MockApiClient:
        mock_client = MockApiClient.return_value
        yield mock_client


@pytest.fixture
def pil_image() -> Image.Image:
    """Create a simple RGB image using PIL."""
    image = Image.new("RGB", (640, 640), color="red")
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


@pytest.fixture
def mock_metadata():
    return ModelMetadata(
        ref="test_model_ref",
        name="Test Model",
        description="A test model for unit tests",
        owner_ref="test_owner",
        focoos_model="test_focoos_model",
        task=FocoosTask.DETECTION,
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        status=ModelStatus.DEPLOYED,
        metrics={"accuracy": 0.95},
        latencies=[{"inference": 0.1}],
        classes=["class_0", "class_1"],
        im_size=640,
        hyperparameters=None,
        training_info=None,
        location=None,
        dataset=None,
    )
