from dataclasses import dataclass
from typing import Optional

import torch

from focoos.ports import ModelOutput


@dataclass
class DETRModelOutput(ModelOutput):
    boxes: torch.Tensor  # [N, num_queries, 4], XYXY normalized to [0, 1]
    logits: torch.Tensor  # [N, num_queries, num_classes]
    loss: Optional[dict]


@dataclass
class DETRTargets:
    labels: torch.Tensor
    boxes: torch.Tensor
