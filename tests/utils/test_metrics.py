from unittest.mock import Mock, patch

import pytest
from colorama import Fore, Style

from focoos.ports import Metrics
from focoos.utils.metrics import MetricsVisualizer


@pytest.fixture
def sample_metrics():
    metrics = Mock(spec=Metrics)
    metrics.train_metrics = [
        {"iteration": 1, "loss_1": 0.5, "total_loss": 1.0},
        {"iteration": 2, "loss_1": 0.3, "total_loss": 0.8},
    ]
    metrics.valid_metrics = [
        {"iteration": 1, "bbox/AP": 0.6, "bbox/AP50": 0.9, "is_valid": True},
        {"iteration": 2, "bbox/AP": 0.7, "bbox/AP50": 0.8, "is_valid": True},
    ]
    metrics.infer_metrics = [
        {"device": "cuda", "engine": "tensorrt", "fps": 30.5},
        {"device": "cuda", "engine": "tensorrt", "fps": 31.0},
    ]
    metrics.best_valid_metric = {"iteration": 2, "bbox/AP": 0.7, "bbox/AP50": 0.8}
    return metrics


@pytest.fixture
def metrics_visualizer(sample_metrics):
    return MetricsVisualizer(sample_metrics)


def test_init(metrics_visualizer):
    """Test the initialization of MetricsVisualizer."""
    assert isinstance(metrics_visualizer, MetricsVisualizer)
    assert metrics_visualizer.green_up == f"{Fore.GREEN}(↑){Style.RESET_ALL}"
    assert metrics_visualizer.red_down == f"{Fore.RED}(↓){Style.RESET_ALL}"


@patch("builtins.print")
def test_log_metrics_training(mock_print, metrics_visualizer):
    """Test logging of training metrics."""
    metrics_visualizer.log_metrics()
    mock_print.assert_any_call(f"{Fore.YELLOW}[Training metrics]{Style.RESET_ALL}")


@patch("builtins.print")
def test_log_metrics_inference(mock_print, metrics_visualizer):
    """Test logging of inference metrics."""
    metrics_visualizer.log_metrics()
    mock_print.assert_any_call(f"{Fore.YELLOW}[Inference metrics]{Style.RESET_ALL}")


def test_notebook_plot_training_metrics(metrics_visualizer):
    """Test plotting functionality."""
    with patch("matplotlib.pyplot.show") as mock_show:
        metrics_visualizer.notebook_plot_training_metrics()
        mock_show.assert_called_once()


def test_notebook_plot_no_metrics():
    """Test plotting behavior when no metrics are available."""
    empty_metrics = Mock(spec=Metrics)
    empty_metrics.train_metrics = []
    empty_metrics.valid_metrics = []
    empty_metrics.infer_metrics = []
    visualizer = MetricsVisualizer(empty_metrics)

    with patch("builtins.print") as mock_print:
        visualizer.notebook_plot_training_metrics()
        mock_print.assert_called_once_with("No training metrics available to plot.")


def test_metrics_comparison(metrics_visualizer):
    """Test metrics comparison and arrow indicators."""
    with patch("builtins.print") as mock_print:
        metrics_visualizer.log_metrics()
        # Verifica che sia stata chiamata almeno una volta print
        assert mock_print.called


@pytest.mark.parametrize(
    "metrics_data",
    [
        {
            "train_metrics": [{"iteration": 1, "loss": 0.5}],
            "valid_metrics": [],
            "infer_metrics": [],
            "best_valid_metric": {"iteration": 1, "loss": 0.5},
        },
        {"train_metrics": [], "valid_metrics": [], "infer_metrics": [{"device": "cuda", "fps": 30.0}]},
    ],
)
def test_metrics_visualizer_edge_cases(metrics_data):
    """Test MetricsVisualizer with various edge cases."""
    metrics = Mock(spec=Metrics)
    for key, value in metrics_data.items():
        setattr(metrics, key, value)
    visualizer = MetricsVisualizer(metrics)
    with patch("builtins.print") as mock_print:
        visualizer.log_metrics()
        assert mock_print.called
