from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

import tests
from focoos.ports import FocoosTask, Hyperparameters, ModelMetadata, TrainInstance
from focoos.remote_model import RemoteModel


def _get_mock_remote_model(
    mocker: MockerFixture, mock_http_client, image_ndarray, mock_metadata: ModelMetadata
):
    mock_http_client.get = MagicMock(
        return_value=MagicMock(status_code=200, json=lambda: mock_metadata.model_dump())
    )
    model = RemoteModel(model_ref="test_model_ref", http_client=mock_http_client)

    # Mock BoxAnnotator
    mock_box_annotator = mocker.patch("focoos.remote_model.BoxAnnotator", autospec=True)
    mock_box_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock LabelAnnotator
    mock_label_annotator = mocker.patch(
        "focoos.remote_model.LabelAnnotator", autospec=True
    )
    mock_label_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock MaskAnnotator
    mock_mask_annotator = mocker.patch(
        "focoos.remote_model.MaskAnnotator", autospec=True
    )
    mock_mask_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    return model


@pytest.fixture
def mock_remote_model(
    mocker: MockerFixture, mock_http_client, image_ndarray, mock_metadata: ModelMetadata
):
    return _get_mock_remote_model(
        mocker=mocker,
        mock_http_client=mock_http_client,
        image_ndarray=image_ndarray,
        mock_metadata=mock_metadata,
    )


def test_remote_model_initialization_fail_to_fetch_model_info(mock_http_client):
    with pytest.raises(ValueError):
        mock_http_client.get = MagicMock(return_value=MagicMock(status_code=500))
        RemoteModel(model_ref="test_model_ref", http_client=mock_http_client)


def test_remote_model_initialization_ok(
    mocker: MockerFixture, mock_http_client, image_ndarray, mock_metadata: ModelMetadata
):
    with tests.not_raises(Exception):
        _get_mock_remote_model(
            mocker=mocker,
            mock_http_client=mock_http_client,
            image_ndarray=image_ndarray,
            mock_metadata=mock_metadata,
        )


def test_train_status_fail(mock_remote_model: RemoteModel):
    with pytest.raises(ValueError):
        mock_remote_model.http_client.get = MagicMock(
            return_value=MagicMock(status_code=500)
        )
        mock_remote_model.train_status()


def test_train_status_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.http_client.get = MagicMock(
            return_value=MagicMock(
                status_code=200, json=MagicMock(return_value={"status": "training"})
            )
        )
        result = mock_remote_model.train_status()
        assert result == {"status": "training"}


def test_train_logs_fail(mock_remote_model: RemoteModel):
    mock_remote_model.http_client.get = MagicMock(
        return_value=MagicMock(status_code=500, text="Internal Server Error")
    )
    result = mock_remote_model.train_logs()
    assert result == []


def test_train_logs_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.http_client.get = MagicMock(
            return_value=MagicMock(
                status_code=200, json=MagicMock(return_value=["log1", "log2"])
            )
        )
        result = mock_remote_model.train_logs()
        assert result == ["log1", "log2"]


def test_stop_training_fail(mock_remote_model: RemoteModel):
    with pytest.raises(ValueError):
        mock_remote_model.http_client.delete = MagicMock(
            return_value=MagicMock(status_code=500)
        )
        mock_remote_model.stop_training()


def test_stop_training_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.http_client.delete = MagicMock(
            return_value=MagicMock(status_code=200)
        )
        mock_remote_model.stop_training()


def test_delete_model_fail(mock_remote_model: RemoteModel):
    with pytest.raises(ValueError):
        mock_remote_model.http_client.delete = MagicMock(
            return_value=MagicMock(status_code=500)
        )
        mock_remote_model.delete_model()


def test_delete_model_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.http_client.delete = MagicMock(
            return_value=MagicMock(status_code=204)
        )
        mock_remote_model.delete_model()


def test_train_metrics_fail(mock_remote_model: RemoteModel):
    mock_remote_model.http_client.get = MagicMock(
        return_value=MagicMock(status_code=500, text="Internal Server Error")
    )
    result = mock_remote_model.train_metrics()
    assert result is None


def test_train_metrics_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.http_client.get = MagicMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"accuracy": 0.95, "loss": 0.1}),
            )
        )
        result = mock_remote_model.train_metrics()
        assert result == {"accuracy": 0.95, "loss": 0.1}


@pytest.fixture()
def mock_hyperparameters(mocker: MockerFixture):
    return mocker.patch(
        "focoos.remote_model.Hyperparameters.model_dump",
        return_value={"learning_rate": 0.01, "batch_size": 32},
    )


def test_train_fail(
    mock_remote_model: RemoteModel,
    mock_hyperparameters: Hyperparameters,
):
    with pytest.raises(ValueError):
        mock_remote_model.http_client.post = MagicMock(
            return_value=MagicMock(status_code=500)
        )
        mock_remote_model.train(
            dataset_ref="dataset_123",
            hyperparameters=mock_hyperparameters,
            anyma_version="anyma-sagemaker-cu12-torch22-0111",
            instance_type=TrainInstance.ML_G4DN_XLARGE,
            volume_size=50,
            max_runtime_in_seconds=36000,
        )


def test_train_ok(
    mock_remote_model: RemoteModel, mock_hyperparameters: Hyperparameters
):
    mock_remote_model.http_client.post = MagicMock(
        return_value=MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "status": "training started",
                    "model_ref": "model_123",
                }
            ),
        )
    )
    result = mock_remote_model.train(
        dataset_ref="dataset_123",
        hyperparameters=mock_hyperparameters,
        anyma_version="anyma-sagemaker-cu12-torch22-0111",
        instance_type=TrainInstance.ML_G4DN_XLARGE,
        volume_size=50,
        max_runtime_in_seconds=36000,
    )
    assert result == {"status": "training started", "model_ref": "model_123"}


def test_log_metrics_semseg(mock_remote_model: RemoteModel, mocker):
    mocker.patch.object(
        mock_remote_model,
        "train_metrics",
        return_value={
            "iter": [1, 2, 3],
            "total_loss": [0.5, 0.4, 0.3],
            "mIoU": [0.7, 0.8, 0.85],
        },
    )
    mock_remote_model.metadata.task = FocoosTask.SEMSEG
    mock_logger = mocker.patch("focoos.remote_model.logger.info")

    mock_remote_model._log_metrics()

    mock_logger.assert_called_once_with("Iter 3: Loss 0.30, mIoU 0.85")


def test_log_metrics_detection(mock_remote_model: RemoteModel, mocker):
    mocker.patch.object(
        mock_remote_model,
        "train_metrics",
        return_value={
            "iter": [1, 2, 3],
            "total_loss": [0.6, 0.5, 0.4],
            "AP50": [0.75, 0.8, 0.82],
        },
    )
    mock_remote_model.metadata.task = FocoosTask.DETECTION
    mock_logger = mocker.patch("focoos.remote_model.logger.info")

    mock_remote_model._log_metrics()

    mock_logger.assert_called_once_with("Iter 3: Loss 0.40, AP50 0.82")


def test_log_metrics_empty_metrics(mock_remote_model: RemoteModel, mocker):
    mocker.patch.object(mock_remote_model, "train_metrics", return_value=None)
    mock_logger = mocker.patch("focoos.remote_model.logger.info")

    mock_remote_model._log_metrics()

    mock_logger.assert_not_called()


def test_log_metrics_missing_keys(mock_remote_model: RemoteModel, mocker):
    mocker.patch.object(
        mock_remote_model,
        "train_metrics",
        return_value={
            "iter": [1, 2, 3],
            # 'total_loss' key is missing
            "AP50": [0.75, 0.8, 0.82],
        },
    )
    mock_remote_model.metadata.task = FocoosTask.DETECTION
    mock_logger = mocker.patch("focoos.remote_model.logger.info")

    mock_remote_model._log_metrics()

    mock_logger.assert_called_once_with("Iter 3: Loss -1.00, AP50 0.82")
