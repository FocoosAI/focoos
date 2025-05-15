from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from focoos.ports import DatasetEntry, LatencyMetrics, ModelConfig, ModelOutput
from focoos.utils.checkpoint import IncompatibleKeys, strip_prefix_if_present
from focoos.utils.logger import get_logger

logger = get_logger("BaseModelNN")


class BaseModelNN(ABC, nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError("Device is not implemented for this model.")

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
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
        raise NotImplementedError("Forward is not implemented for this model.")

    def load_state_dict(self, checkpoint_state_dict: dict, strict: bool = True) -> IncompatibleKeys:
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

    def benchmark(self, iterations: int = 50, size: int = 640) -> LatencyMetrics:
        try:
            model = self.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        logger.info(f"⏱️ Benchmarking latency on {model.device}, size: {size}x{size}..")
        # warmup
        data = 128 * torch.randn(1, 3, size, size).to(model.device)

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
        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=f"torch.{self.device}",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size,
            device=str(self.device),
        )
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics
