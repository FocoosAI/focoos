from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from focoos.ports import LatencyMetrics, RemoteModelInfo


class BaseRuntime:
    """
    Abstract base class for runtime implementations.

    This class defines the interface that all runtime implementations must follow.
    It provides methods for model initialization, inference, and performance benchmarking.

    Attributes:
        model_path (str): Path to the model file.
        opts (Any): Runtime-specific options.
        model_info (RemoteModelInfo): Metadata about the model.
    """

    def __init__(self, model_path: str, opts: Any, model_info: RemoteModelInfo):
        """
        Initialize the runtime with model path, options and metadata.

        Args:
            model_path (str): Path to the model file.
            opts (Any): Runtime-specific configuration options.
            model_info (RemoteModelInfo): Metadata about the model.
        """
        pass

    @abstractmethod
    def __call__(self, im: torch.Tensor) -> list[np.ndarray]:
        """
        Run inference on the input image.

        Args:
            im (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Model output as a numpy array.
        """
        pass

    @abstractmethod
    def get_info(self) -> tuple[str, str]:
        """
        Get the engine and device name.
        """
        pass

    @abstractmethod
    def benchmark(self, iterations: int, size: float) -> LatencyMetrics:
        """
        Benchmark the model performance.

        Args:
            iterations (int): Number of inference iterations to run.
            size (float): Input image size for benchmarking.

        Returns:
            LatencyMetrics: Performance metrics including mean, median, and percentile latencies.
        """
        pass
