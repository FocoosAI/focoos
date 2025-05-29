import pathlib
import tempfile
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from focoos import FocoosHUB
from focoos.config import FOCOOS_CONFIG
from focoos.hub.remote_model import RemoteModel
from focoos.infer.infer_model import InferModel
from focoos.ports import ModelPreview


@pytest.fixture
def focoos_instance(mock_api_client) -> FocoosHUB:
    """Fixture to provide a Focoos instance with a mocked ApiClient."""
    mock_api_client.get.return_value.status_code = 200
    mock_api_client.get.return_value.json.return_value = {
        "email": "test@example.com",
        "created_at": "2024-01-01",
        "updated_at": "2025-01-01",
        "company": "test_company",
        "api_key": {"key": "test_api_key"},
        "quotas": {
            "total_inferences": 10,
            "max_inferences": 1000,
            "used_storage_gb": 10,
            "max_storage_gb": 1000,
            "active_training_jobs": ["job1"],
            "max_active_training_jobs": 1,
            "used_mlg4dnxlarge_training_jobs_hours": 10,
            "max_mlg4dnxlarge_training_jobs_hours": 1000,
        },
    }
    return FocoosHUB(api_key="test_api_key", host_url="http://mock-host-url.com")


@pytest.fixture
def mock_shared_datasets():
    return [
        {
            "ref": "79742c8814f94fcd",
            "url": "s3://mock-s3-url",
            "name": "Aeroscapes",
            "layout": "supervisely",
            "description": "Mock descr 1",
            "task": "semseg",
        },
        {
            "ref": "cce71b2050be4e28",
            "url": "s3://mock-s3-url",
            "name": "Blister",
            "layout": "roboflow_coco",
            "description": "Mock descr 2",
            "task": "instseg",
        },
    ]


@pytest.fixture
def mock_list_models():
    return [
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


@pytest.fixture
def mock_list_models_as_base_models(mock_list_models):
    return [ModelPreview.from_json(r) for r in mock_list_models]


@pytest.fixture
def mock_remote_model():
    return MagicMock(spec=RemoteModel, model_ref="ref1")


@pytest.fixture
def mock_local_model():
    return MagicMock(spec=InferModel, name="model1", model_ref="ref1")


def test_focoos_initialization_no_api_key(focoos_instance: FocoosHUB):
    focoos_instance.api_client.get = MagicMock(
        return_value=MagicMock(status_code=200, json=lambda: {"email": "test@example.com"})
    )
    FOCOOS_CONFIG.focoos_api_key = ""
    with pytest.raises(ValueError):
        FocoosHUB(host_url="http://mock-host-url.com")


def test_focoos_initialization_fail_to_fetch_user_info(focoos_instance: FocoosHUB):
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=500))
    with pytest.raises(ValueError):
        FocoosHUB(api_key="test_api_key")


def test_focoos_initialization(focoos_instance: FocoosHUB):
    assert focoos_instance.api_key == "test_api_key"
    assert focoos_instance.user_info.email == "test@example.com"
    assert focoos_instance.user_info.company == "test_company"
    assert focoos_instance.user_info.created_at == datetime(2024, 1, 1)
    assert focoos_instance.user_info.updated_at == datetime(2025, 1, 1)
    assert focoos_instance.user_info.quotas.total_inferences == 10
    assert focoos_instance.user_info.quotas.max_inferences == 1000
    assert focoos_instance.user_info.quotas.used_storage_gb == 10
    assert focoos_instance.user_info.quotas.max_storage_gb == 1000
    assert focoos_instance.user_info.quotas.active_training_jobs == ["job1"]
    assert focoos_instance.user_info.quotas.max_active_training_jobs == 1
    assert focoos_instance.user_info.quotas.used_mlg4dnxlarge_training_jobs_hours == 10
    assert focoos_instance.user_info.quotas.max_mlg4dnxlarge_training_jobs_hours == 1000


def test_get_model_info(focoos_instance: FocoosHUB):
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
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: mock_response))
    model_info = focoos_instance.get_model_info("test-model")
    assert model_info.name == "test-model"
    assert model_info.ref == "model-ref"
    assert model_info.description == "Test model description"


def test_get_model_info_fail(focoos_instance: FocoosHUB):
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=500))

    with pytest.raises(ValueError):
        focoos_instance.get_model_info("test-model")


def test_list_models(focoos_instance: FocoosHUB, mock_list_models):
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: mock_list_models))

    models = focoos_instance.list_remote_models()
    assert len(models) == 2
    assert models[0].name == "model1"
    assert models[1].ref == "ref2"


def test_list_models_fail(focoos_instance: FocoosHUB):
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=500))

    with pytest.raises(ValueError):
        focoos_instance.list_remote_models()


def test_list_shared_datasets(focoos_instance: FocoosHUB, mock_shared_datasets):
    focoos_instance.api_client.get = MagicMock(
        return_value=MagicMock(status_code=200, json=lambda: mock_shared_datasets)
    )

    res = focoos_instance.list_shared_datasets()

    assert len(res) == 2
    assert res[0].name == "Aeroscapes"
    assert res[1].ref == "cce71b2050be4e28"


def test_list_shared_datasets_fail(focoos_instance: FocoosHUB):
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=500))
    with pytest.raises(ValueError):
        focoos_instance.list_shared_datasets()


"""
unit tests get_model_by_name
"""


def test_get_remote_model(mocker: MockerFixture, focoos_instance: FocoosHUB, mock_remote_model, mock_api_client):
    mock_remote_model_class = mocker.patch("focoos.focoos.RemoteModel", autospec=True)
    mock_remote_model_class.return_value = mock_remote_model
    model_ref = "ref1"
    model = focoos_instance.get_remote_model(model_ref)
    assert model is not None
    assert model.model_ref == model_ref
    mock_remote_model_class.assert_called_once_with(model_ref, mock_api_client)
    assert isinstance(model, RemoteModel)


def test_new_model_created(
    mocker: MockerFixture,
    focoos_instance: FocoosHUB,
    mock_remote_model: RemoteModel,
    mock_api_client,
):
    focoos_instance.api_client.post = MagicMock(
        return_value=MagicMock(
            status_code=201,
            json=lambda: {
                "ref": mock_remote_model.model_ref,
            },
        )
    )
    mock_remote_model_class = mocker.patch("focoos.focoos.RemoteModel", autospec=True)
    mock_remote_model_class.return_value = mock_remote_model

    model = focoos_instance.new_model("fakename", "fakefocoosmodel", "fakedescription")

    assert model is not None
    mock_remote_model_class.assert_called_once_with(mock_remote_model.model_ref, mock_api_client)
    assert isinstance(model, RemoteModel)


def test_new_model_already_exists(mocker: MockerFixture, focoos_instance: FocoosHUB, mock_remote_model: RemoteModel):
    model_name = "fakename"
    focoos_instance.api_client.post = MagicMock(return_value=MagicMock(status_code=409))
    mock_get_model_by_name = mocker.patch.object(focoos_instance, "get_model_by_name", autospec=True)
    mock_get_model_by_name.return_value = mock_remote_model

    model = focoos_instance.new_model(model_name, "fakefocoosmodel", "fakedescription")
    assert model is not None
    mock_get_model_by_name.assert_called_once_with(model_name, remote=True)
    assert isinstance(model, RemoteModel)


def test_new_model_fail(focoos_instance: FocoosHUB):
    model_name = "fakename"
    focoos_instance.api_client.post = MagicMock(return_value=MagicMock(status_code=500))
    model = focoos_instance.new_model(model_name, "fakefocoosmodel", "fakedescription")
    assert model is None


def test_download_model_already_exists(focoos_instance: FocoosHUB):
    model_ref = "ref1"
    with tempfile.TemporaryDirectory() as model_dir_tmp:
        focoos_instance.focoos_dir = model_dir_tmp
        model_dir_tmp = pathlib.Path(model_dir_tmp) / model_ref
        model_dir_tmp.mkdir(parents=True, exist_ok=True)
        model_onnx_path = model_dir_tmp / "model.onnx"
        model_onnx_path.touch()
        (model_dir_tmp / "focoos_metadata.json").touch()
        model_path = focoos_instance._download_model(model_ref)
        assert model_path is not None
        assert model_path == str(model_dir_tmp / "model.onnx")


def test_download_model_onnx_fail(focoos_instance: FocoosHUB):
    model_ref = "ref1"
    focoos_instance.api_client.get = MagicMock(return_value=MagicMock(status_code=500))
    with tempfile.TemporaryDirectory() as model_dir_tmp:
        focoos_instance.focoos_dir = model_dir_tmp
        with pytest.raises(ValueError):
            focoos_instance._download_model(model_ref)
        assert not (pathlib.Path(focoos_instance.focoos_dir) / "model.onnx").exists()


def test_download_model_onnx_ok_but_get_external_fail(mocker: MockerFixture, focoos_instance: FocoosHUB):
    model_ref = "ref1"
    # Mock successful API response for model metadata
    focoos_instance.api_client.get = MagicMock(
        return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "download_uri": "https://fake.com",
                "model_metadata": MagicMock(),
            },
        ),
    )
    # Mock model metadata parsing
    mock_model_metadata = mocker.patch("focoos.focoos.ModelMetadata.from_json", autospec=True)
    mock_model_metadata.return_value = MagicMock(model_dump_json=lambda: "fake_model_dump")

    with tempfile.TemporaryDirectory() as model_dir_tmp:
        focoos_instance.focoos_dir = model_dir_tmp
        # Mock failed download from Focoos Cloud
        focoos_instance.api_client.download_ext_file = MagicMock(side_effect=ValueError("Failed to download model"))

        # Should raise ValueError when download fails
        with pytest.raises(ValueError, match="Failed to download model"):
            focoos_instance._download_model(model_ref)

        # Verify no files were created
        model_dir = pathlib.Path(model_dir_tmp) / model_ref
        assert not (model_dir / "model.onnx").exists()
        assert not (model_dir / "focoos_metadata.json").exists()


def test_download_model_onnx(mocker: MockerFixture, focoos_instance: FocoosHUB):
    with tempfile.TemporaryDirectory() as model_dir_tmp:
        focoos_instance.focoos_dir = model_dir_tmp
        model_ref = "ref1"
        expected_path = str(pathlib.Path(model_dir_tmp) / model_ref / "model.onnx")
        focoos_instance.api_client.get = MagicMock(
            return_value=MagicMock(
                status_code=200,
                json=lambda: {
                    "download_uri": "https://fake.com",
                    "model_metadata": MagicMock(),
                },
            ),
        )
        focoos_instance.api_client.external_get = MagicMock(return_value=MagicMock(status_code=200))
        focoos_instance.api_client.download_ext_file = MagicMock(return_value=expected_path)
        mock_model_metadata = mocker.patch("focoos.focoos.ModelMetadata.from_json", autospec=True)
        mock_model_metadata.return_value = MagicMock(model_dump_json=lambda: "fake_model_dump")
        focoos_instance.api_client.external_get = MagicMock(
            return_value=MagicMock(
                status_code=200,
                headers={"content-length": 100},
                iter_content=lambda chunk_size: [
                    b"chunk1",
                    b"chunk2",
                    b"chunk3",
                    b"chunk4",
                ],
            )
        )

        model_path = focoos_instance._download_model(model_ref)
        assert model_path is not None
        assert model_path == expected_path
