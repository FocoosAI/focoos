from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from focoos.ports import DatasetEntry, ModelConfig, ModelOutput
from focoos.utils.checkpoint import IncompatibleKeys, strip_prefix_if_present


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
