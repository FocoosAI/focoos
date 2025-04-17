from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

import tests
from focoos.ports import Hyperparameters, Metrics, ModelStatus, RemoteModelInfo, Task, TrainingInfo, TrainInstance
from focoos.remote_model import RemoteModel


def _get_mock_remote_model(mocker: MockerFixture, mock_api_client, image_ndarray, mock_metadata: RemoteModelInfo):
    mock_api_client.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: mock_metadata.model_dump()))
    model = RemoteModel(model_ref="test_model_ref", api_client=mock_api_client)

    # Mock BoxAnnotator
    mock_box_annotator = mocker.patch("focoos.remote_model.sv.BoxAnnotator", autospec=True)
    mock_box_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock LabelAnnotator
    mock_label_annotator = mocker.patch("focoos.remote_model.sv.LabelAnnotator", autospec=True)
    mock_label_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    # Mock MaskAnnotator
    mock_mask_annotator = mocker.patch("focoos.remote_model.sv.MaskAnnotator", autospec=True)
    mock_mask_annotator.annotate = MagicMock(return_value=np.zeros_like(image_ndarray))

    return model


@pytest.fixture
def mock_remote_model(mocker: MockerFixture, mock_api_client, image_ndarray, mock_metadata: RemoteModelInfo):
    return _get_mock_remote_model(
        mocker=mocker,
        mock_api_client=mock_api_client,
        image_ndarray=image_ndarray,
        mock_metadata=mock_metadata,
    )


def test_remote_model_initialization_fail_to_fetch_model_info(mock_api_client):
    with pytest.raises(ValueError):
        mock_api_client.get = MagicMock(return_value=MagicMock(status_code=500))
        RemoteModel(model_ref="test_model_ref", api_client=mock_api_client)


def test_remote_model_initialization_ok(
    mocker: MockerFixture, mock_api_client, image_ndarray, mock_metadata: RemoteModelInfo
):
    with tests.not_raises(Exception):
        _get_mock_remote_model(
            mocker=mocker,
            mock_api_client=mock_api_client,
            image_ndarray=image_ndarray,
            mock_metadata=mock_metadata,
        )


def test_train_status_fail(mock_remote_model: RemoteModel):
    with pytest.raises(ValueError):
        mock_remote_model.api_client.get = MagicMock(return_value=MagicMock(status_code=500))
        mock_remote_model.train_info()


def test_train_status_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.api_client.get = MagicMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={
                        "algorithm_name": "anyma0.12.7",
                        "instance_type": "ml.g4dn.xlarge",
                        "volume_size": 100,
                    }
                ),
            )
        )
        result = mock_remote_model.train_info()
        assert result == TrainingInfo(
            algorithm_name="anyma0.12.7",
            instance_type="ml.g4dn.xlarge",
            volume_size=100,
        )


def test_train_logs_fail(mock_remote_model: RemoteModel):
    mock_remote_model.api_client.get = MagicMock(return_value=MagicMock(status_code=500, text="Internal Server Error"))
    result = mock_remote_model.train_logs()
    assert result == []


def test_train_logs_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.api_client.get = MagicMock(
            return_value=MagicMock(status_code=200, json=MagicMock(return_value=["log1", "log2"]))
        )
        result = mock_remote_model.train_logs()
        assert result == ["log1", "log2"]


def test_stop_training_fail(mock_remote_model: RemoteModel):
    with pytest.raises(ValueError):
        mock_remote_model.api_client.delete = MagicMock(return_value=MagicMock(status_code=500))
        mock_remote_model.stop_training()


def test_stop_training_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.api_client.delete = MagicMock(return_value=MagicMock(status_code=200))
        mock_remote_model.stop_training()


def test_delete_model_fail(mock_remote_model: RemoteModel):
    with pytest.raises(ValueError):
        mock_remote_model.api_client.delete = MagicMock(return_value=MagicMock(status_code=500))
        mock_remote_model.delete_model()


def test_delete_model_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.api_client.delete = MagicMock(return_value=MagicMock(status_code=204))
        mock_remote_model.delete_model()


def test_train_metrics_fail(mock_remote_model: RemoteModel):
    mock_remote_model.api_client.get = MagicMock(return_value=MagicMock(status_code=500, text="Internal Server Error"))
    result = mock_remote_model.metrics()
    assert result == Metrics()


def test_train_metrics_ok(mock_remote_model: RemoteModel):
    with tests.not_raises(Exception):
        mock_remote_model.api_client.get = MagicMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"train_metrics": [{"iteration": 1, "loss": 0.1, "sem_seg/mIoU": 0.95}]}),
            )
        )
        result = mock_remote_model.metrics()
        assert result == Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.1, "sem_seg/mIoU": 0.95}],
        )


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
        mock_remote_model.api_client.post = MagicMock(return_value=MagicMock(status_code=500))
        mock_remote_model.train(
            dataset_ref="dataset_123",
            hyperparameters=mock_hyperparameters,
            instance_type=TrainInstance.ML_G4DN_XLARGE,
            volume_size=50,
            max_runtime_in_seconds=36000,
        )


def test_train_ok(mock_remote_model: RemoteModel, mock_hyperparameters: Hyperparameters):
    mock_remote_model.api_client.post = MagicMock(
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
        instance_type=TrainInstance.ML_G4DN_XLARGE,
        volume_size=50,
        max_runtime_in_seconds=36000,
    )
    assert result == {"status": "training started", "model_ref": "model_123"}


def test_metrics_semseg(mock_remote_model: RemoteModel, mocker):
    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 2, "loss": 0.01, "sem_seg/mIoU": 0.90}],
            valid_metrics=[{"iteration": 3, "loss": 0.1, "sem_seg/mIoU": 0.95}],
            best_valid_metric={"iteration": 3, "loss": 0.1, "sem_seg/mIoU": 0.95},
        ),
    )
    mock_remote_model.metadata.task = Task.SEMSEG

    metrics = mock_remote_model.metrics()
    assert isinstance(metrics, Metrics)
    assert metrics.best_valid_metric == {"iteration": 3, "loss": 0.1, "sem_seg/mIoU": 0.95}
    assert metrics.train_metrics == [{"iteration": 2, "loss": 0.01, "sem_seg/mIoU": 0.90}]
    assert metrics.valid_metrics == [{"iteration": 3, "loss": 0.1, "sem_seg/mIoU": 0.95}]
    assert metrics.infer_metrics == []


def test_metrics_detection(mock_remote_model: RemoteModel, mocker):
    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.6, "bbox/AP50": 0.75}],
            valid_metrics=[{"iteration": 1, "loss": 0.5, "bbox/AP50": 0.8}],
            best_valid_metric={"iteration": 1, "loss": 0.4, "bbox/AP50": 0.82},
        ),
    )
    mock_remote_model.metadata.task = Task.DETECTION

    metrics = mock_remote_model.metrics()
    assert metrics.best_valid_metric == {"iteration": 1, "loss": 0.4, "bbox/AP50": 0.82}
    assert metrics.train_metrics == [{"iteration": 1, "loss": 0.6, "bbox/AP50": 0.75}]
    assert metrics.valid_metrics == [{"iteration": 1, "loss": 0.5, "bbox/AP50": 0.8}]
    assert metrics.infer_metrics == []


def test_notebook_monitor_train_invalid_interval(mock_remote_model: RemoteModel):
    """Test that monitor_train raises ValueError for invalid intervals."""
    with pytest.raises(ValueError, match="Interval must be between 30 and 240 seconds"):
        mock_remote_model.notebook_monitor_train(interval=20)
    with pytest.raises(ValueError, match="Interval must be between 30 and 240 seconds"):
        mock_remote_model.notebook_monitor_train(interval=250)


def test_notebook_monitor_train_completed(mock_remote_model: RemoteModel, mocker):
    """Test monitoring when training completes successfully."""
    # Mock time functions
    mocker.patch("time.time", return_value=1000)
    mock_sleep = mocker.patch("time.sleep")
    mock_clear = mocker.patch("IPython.display.clear_output")

    # Mock model info and metrics
    mocker.patch.object(
        mock_remote_model, "get_info", return_value=mocker.Mock(status=ModelStatus.TRAINING_COMPLETED, updated_at=1000)
    )
    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.6}],
            valid_metrics=[{"iteration": 1, "loss": 0.5}],
            best_valid_metric={"iteration": 1, "loss": 0.5},
            infer_metrics=[],
        ),
    )

    mock_remote_model.notebook_monitor_train(interval=30, plot_metrics=False)

    assert mock_clear.called
    assert not mock_sleep.called


def test_notebook_monitor_train_running(mock_remote_model: RemoteModel, mocker):
    """Test monitoring during active training."""
    # Mock time functions
    mocker.patch(
        "time.time",
        side_effect=[1000 + i * 30 for i in range(10)],  # Provide enough values for all time.time() calls
    )
    mock_sleep = mocker.patch("focoos.remote_model.sleep")
    # mock_time = mocker.patch("focoos.remote_model.time")
    mock_clear = mocker.patch("IPython.display.clear_output")

    # Mock model info with different states

    mocker.patch.object(
        mock_remote_model,
        "get_info",
        side_effect=[
            mocker.Mock(status=ModelStatus.TRAINING_RUNNING, updated_at=1001),
            mocker.Mock(status=ModelStatus.TRAINING_RUNNING, updated_at=1002),
            mocker.Mock(status=ModelStatus.TRAINING_COMPLETED, updated_at=1003),
        ],
    )
    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.6}],
            valid_metrics=[{"iteration": 1, "loss": 0.5}],
            best_valid_metric={"iteration": 1, "loss": 0.5, "sem_seg/mIoU": 0.95},
            infer_metrics=[],
        ),
    )

    mock_remote_model.notebook_monitor_train(interval=30, plot_metrics=False)
    print(f"sleep call count: {mock_sleep.call_count}")
    assert mock_clear.called
    assert mock_clear.call_count == 2
    assert mock_sleep.call_count == 1

    assert mock_sleep.call_args == mocker.call(30)


def test_notebook_monitor_train_max_runtime(mock_remote_model: RemoteModel, mocker):
    """Test that monitoring stops when max runtime is exceeded."""
    # Mock time module
    mocker.patch(
        "focoos.remote_model.time",
        **{
            "time": mocker.Mock(side_effect=[1000, 40000]),  # First call for start_time, second for check
            "sleep": mocker.Mock(),  # Prevent actual sleeping
        },
    )

    mock_clear = mocker.patch("IPython.display.clear_output")
    mock_logger = mocker.patch("logging.Logger.warning")

    # Create a mock that always returns TRAINING_RUNNING
    mock_info = mocker.Mock()
    mock_info.status = ModelStatus.TRAINING_RUNNING
    mock_info.updated_at = 1000

    mocker.patch.object(
        mock_remote_model,
        "get_info",
        return_value=mock_info,  # Always return the same mock object
    )

    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.6}],
            valid_metrics=[],
            best_valid_metric=None,
            infer_metrics=[],
        ),
    )

    mock_remote_model.notebook_monitor_train(interval=30, max_runtime=36000)
    assert mock_clear.called

    # Verify that warning was logged
    mock_logger.assert_called_with("Monitoring exceeded 36000 seconds limit")


def test_notebook_monitor_train_with_metrics_plot(mock_remote_model: RemoteModel, mocker):
    """Test monitoring with metrics plotting enabled."""
    mocker.patch("time.time", return_value=1000)
    mock_clear = mocker.patch("IPython.display.clear_output")
    mock_plot = mocker.patch("focoos.utils.metrics.MetricsVisualizer.notebook_plot_training_metrics")

    mocker.patch.object(
        mock_remote_model,
        "get_info",
        return_value=mocker.Mock(status=ModelStatus.TRAINING_COMPLETED, updated_at=1000),
    )
    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.6}],
            valid_metrics=[{"iteration": 1, "loss": 0.5}],
            best_valid_metric={"iteration": 1, "loss": 0.5},
            infer_metrics=[],
        ),
    )
    mocker.patch.object(
        mock_remote_model,
        "metrics",
        return_value=Metrics(
            train_metrics=[{"iteration": 1, "loss": 0.6}],
            valid_metrics=[{"iteration": 1, "loss": 0.5}],
            best_valid_metric={"iteration": 1, "loss": 0.5, "sem_seg/mIoU": 0.95},
            infer_metrics=[],
        ),
    )

    mock_remote_model.notebook_monitor_train(interval=30, plot_metrics=True)

    assert mock_plot.called
    assert mock_clear.called


def test_notebook_monitor_train_status_history(mock_remote_model: RemoteModel, mocker):
    """Test status history logging during monitoring."""
    mocker.patch("time.time", return_value=1000)
    mocker.patch("IPython.display.clear_output")
    mock_logger = mocker.patch("logging.Logger.info")

    mocker.patch.object(
        mock_remote_model,
        "get_info",
        return_value=mocker.Mock(status=ModelStatus.TRAINING_COMPLETED, updated_at=1000),
    )
    mock_remote_model.metadata.name = "test_model"

    mock_remote_model.notebook_monitor_train(interval=30)

    expected_msg = f"[Live Monitor test_model] {ModelStatus.TRAINING_COMPLETED.value}"
    mock_logger.assert_any_call(expected_msg)
