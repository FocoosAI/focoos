from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from focoos.ports import DatasetEntry, LatencyMetrics, ModelConfig, ModelOutput
from focoos.utils.checkpoint import IncompatibleKeys, strip_prefix_if_present
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_device_name

logger = get_logger("BaseModelNN")


class BaseModelNN(ABC, nn.Module):
    """Abstract base class for neural network models in Focoos.

    This class provides a common interface for all neural network models,
    defining abstract methods that must be implemented by concrete model classes.
    It extends both ABC (Abstract Base Class) and nn.Module from PyTorch.

    Args:
        config: Model configuration containing hyperparameters and settings.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the base model.

        Args:
            config: Model configuration object containing model parameters
                and settings.
        """
        super().__init__()

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device where the model is located.

        Returns:
            The PyTorch device (CPU or CUDA) where the model parameters
            are stored.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Device is not implemented for this model.")

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Get the data type of the model parameters.

        Returns:
            The PyTorch data type (e.g., float32, float16) of the model
            parameters.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Dtype is not implemented for this model.")

    @abstractmethod
    def forward(
        self,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
            list[DatasetEntry],
        ],
    ) -> ModelOutput:
        """Perform forward pass through the model.

        Args:
            inputs: Input data in various supported formats:
                - torch.Tensor: Single tensor input
                - np.ndarray: Single numpy array input
                - Image.Image: Single PIL Image input
                - list[Image.Image]: List of PIL Images
                - list[np.ndarray]: List of numpy arrays
                - list[torch.Tensor]: List of tensors
                - list[DatasetEntry]: List of dataset entries

        Returns:
            Model output containing predictions and any additional metadata.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Forward is not implemented for this model.")

    def load_state_dict(self, checkpoint_state_dict: dict, strict: bool = True) -> IncompatibleKeys:
        """Load model state dictionary from checkpoint with preprocessing.

        This method handles common issues when loading checkpoints:
        - Removes "module." prefix from DataParallel/DistributedDataParallel models
        - Handles shape mismatches by removing incompatible parameters
        - Logs incompatible keys for debugging

        Args:
            checkpoint_state_dict: Dictionary containing model parameters from
                a saved checkpoint.
            strict: Whether to strictly enforce that the keys in checkpoint_state_dict
                match the keys returned by this module's state_dict() function.
                Defaults to True.

        Returns:
            IncompatibleKeys object containing information about missing keys,
            unexpected keys, and parameters with incorrect shapes.
        """
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)

        incompatible = super().load_state_dict(checkpoint_state_dict, strict=strict)
        incompatible = IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

        incompatible.log_incompatible_keys()

        return incompatible

    def benchmark(self, iterations: int = 50, size: Tuple[int, int] = (640, 640)) -> LatencyMetrics:
        """Benchmark model inference latency and throughput.

        Performs multiple inference runs on random data to measure model
        performance metrics including FPS, mean latency, and latency statistics.
        Uses CUDA events for precise timing when running on GPU.

        Args:
            iterations: Number of inference runs to perform for benchmarking.
                Defaults to 50.
            size: Input image size as (height, width) tuple. Defaults to (640, 640).

        Returns:
            LatencyMetrics object containing:
                - fps: Frames per second (throughput)
                - engine: Hardware/framework used for inference
                - mean: Mean inference time in milliseconds
                - max: Maximum inference time in milliseconds
                - min: Minimum inference time in milliseconds
                - std: Standard deviation of inference times
                - im_size: Input image size
                - device: Device used for inference

        Note:
            This method assumes the model is running on CUDA for timing.
            Input data is randomly generated for benchmarking purposes.
        """
        if self.device.type == "cpu":
            device_name = get_cpu_name()
        else:
            device_name = get_device_name()

        # Normalize size to tuple format
        if isinstance(size, int):
            size_tuple = (size, size)
            size_str = f"{size}x{size}"
        else:
            size_tuple = size
            size_str = f"{size[0]}x{size[1]}"

        logger.info(f"⏱️ Benchmarking latency on {device_name} ({self.device}), size: {size_str}..")
        # warmup
        data = 128 * torch.randn(1, 3, size_tuple[0], size_tuple[1]).to(self.device)
        durations = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream=torch.cuda.Stream())
            _ = self(data)
            end.record(stream=torch.cuda.Stream())
            torch.cuda.synchronize()
            durations.append(start.elapsed_time(end))

        durations = np.array(durations)
        # For LatencyMetrics.im_size (int), use height (first dimension) as representative value
        # This maintains backward compatibility while supporting non-square images
        im_size_repr = size_tuple[0] if isinstance(size, tuple) and size_tuple[0] != size_tuple[1] else size_tuple[0]
        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=f"torch.{self.device}",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=im_size_repr,
            device=device_name,
        )
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics

    def switch_to_export(self, test_cfg: Optional[dict] = None, device: str = "cuda"):
        pass
