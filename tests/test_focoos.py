import pytest
from unittest.mock import patch, MagicMock
from focoos.ports import ModelMetadata, DatasetMetadata, ModelPreview
from focoos.focoos import Focoos
from focoos.utils.system import HttpClient


@pytest.fixture
def focoos_instance():
    with patch("focoos.focoos.HttpClient") as MockHttpClient:
        mock_client = MockHttpClient.return_value
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.json.return_value = {"email": "test@example.com"}
        return Focoos(api_key="test_api_key", host_url="http://mock-host-url.com")


def test_focoos_initialization(focoos_instance):
    assert focoos_instance.api_key == "test_api_key"
    assert focoos_instance.user_info["email"] == "test@example.com"


def test_get_model_info(focoos_instance):
    mock_response = {
        "name": "test-model",
        "ref": "model-ref",
        "owner_ref": "pytest",
        "focoos_model": "focoos_rtdetr",
        "created_at": "2024-01-01",
        "updated_at": "2024-01-01",
        "description": "Test model description",
        "task": "detection",
        "status": "TRAINING_COMPLETED",
    }
    focoos_instance.http_client.get = MagicMock(
        return_value=MagicMock(status_code=200, json=lambda: mock_response)
    )
    model_info = focoos_instance.get_model_info("test-model")
    assert model_info.name == "test-model"
    assert model_info.ref == "model-ref"
    assert model_info.description == "Test model description"


def test_list_models(focoos_instance):
    mock_response = [
        {
            "name": "model1",
            "ref": "ref1",
            "task": "detection",
            "description": "model1 description",
            "status": "TRAINING_COMPLETED",
            "focoos_model": "focoos_rtdetr",
        },
        {
            "name": "model2",
            "ref": "ref2",
            "task": "detection",
            "description": "model2 description",
            "status": "TRAINING_RUNNING",
            "focoos_model": "focoos_rtdetr",
        },
    ]

    focoos_instance.http_client.get = MagicMock(
        return_value=MagicMock(status_code=200, json=lambda: mock_response)
    )

    models = focoos_instance.list_models()
    assert len(models) == 2
    assert models[0].name == "model1"
    assert models[1].ref == "ref2"
