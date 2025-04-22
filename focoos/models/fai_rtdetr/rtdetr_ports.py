from dataclasses import dataclass

import torch

from focoos.ports import ModelOutput


@dataclass
class RTDETRModelOutput(ModelOutput):
    boxes: torch.Tensor  # [N, num_queries, 4], XYXY normalized to [0, 1]
    logits: torch.Tensor  # [N, num_queries, num_classes]


@dataclass
class RTDETRTargets:
    labels: torch.Tensor
    boxes: torch.Tensor
