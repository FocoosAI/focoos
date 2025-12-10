from time import perf_counter
from typing import Literal, Tuple, Union

import numpy as np
import torch

from focoos.infer.runtimes.base import BaseRuntime
from focoos.ports import LatencyMetrics, ModelInfo, Task, TorchscriptRuntimeOpts
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_device_name, get_device_type

logger = get_logger("TorchscriptRuntime")


class TorchscriptRuntime(BaseRuntime):
    """
    TorchScript Runtime wrapper for model inference.

    This class implements the BaseRuntime interface for TorchScript models,
    supporting both CPU and CUDA devices. It handles model initialization,
    device placement, warmup, inference, and performance benchmarking.

    Attributes:
        device (torch.device): Device to run inference on (CPU or CUDA).
        opts (TorchscriptRuntimeOpts): Configuration options for the TorchScript runtime.
        model (torch.jit.ScriptModule): Loaded TorchScript model.
        model_info (RemoteModelInfo): Metadata about the model.
    """

    def __init__(
        self,
        model_path: str,
        opts: TorchscriptRuntimeOpts,
        model_info: ModelInfo,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        if device == "auto":
            self.device = torch.device(get_device_type())
        else:
            self.device = torch.device(device)
        logger.info(f"🔧 Device: {self.device}")
        self.opts = opts
        self.model_info = model_info

        map_location = None if torch.cuda.is_available() else "cpu"

        self.model = torch.jit.load(model_path, map_location=map_location)
        self.model = self.model.to(self.device)

        if self.opts.warmup_iter > 0:
            size = (
                self.model_info.im_size
                if self.model_info.task in [Task.DETECTION, Task.CLASSIFICATION] and self.model_info.im_size
                else 640
            )
            # Normalize size to tuple format
            if isinstance(size, int):
                size_tuple = (size, size)
                size_str = f"{size}x{size}"
            else:
                size_tuple = size
                size_str = f"{size[0]}x{size[1]}"

            logger.info(f"⏱️ Warming up model {self.model_info.name} on {self.device}, size: {size_str}..")
            with torch.no_grad():
                np_image = torch.rand(1, 3, size_tuple[0], size_tuple[1]).to(self.device)
                for _ in range(self.opts.warmup_iter):
                    self.model(np_image)

    def __call__(self, im: torch.Tensor) -> list[np.ndarray]:
        """
        Run inference on the input image.

        Args:
            im (np.ndarray): Input image as a numpy array.

        Returns:
            list[np.ndarray]: Model outputs as a list of numpy arrays.
        """
        with torch.no_grad():
            res = self.model(im)
            return res

    def benchmark(self, iterations: int = 20, size: Union[int, Tuple[int, int]] = 640) -> LatencyMetrics:
        """
        Benchmark the model performance.

        Runs multiple inference iterations and measures execution time to calculate
        performance metrics like FPS, mean latency, and other statistics.

        Args:
            iterations (int, optional): Number of inference iterations to run. Defaults to 20.

        Returns:
            LatencyMetrics: Performance metrics including FPS, mean, min, max, and std latencies.
        """
        engine = "torchscript"
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

        logger.info(f"⏱️ Benchmarking latency on {device_name}, size: {size_str}..")

        torch_input = torch.rand(1, 3, size_tuple[0], size_tuple[1], device=self.device)
        durations = []

        with torch.no_grad():
            for step in range(iterations + 5):
                start = perf_counter()
                self.model(torch_input)
                end = perf_counter()

                if step >= 5:  # Skip first 5 iterations
                    durations.append((end - start) * 1000)

        durations = np.array(durations)

        # For LatencyMetrics.im_size (int), use height (first dimension) as representative value
        im_size_repr = size_tuple[0] if isinstance(size, tuple) and size_tuple[0] != size_tuple[1] else size_tuple[0]
        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean().astype(float)),
            engine=engine,
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=im_size_repr,
            device=device_name,
        )
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics
