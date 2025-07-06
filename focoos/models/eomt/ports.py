from dataclasses import dataclass
from typing import Optional

import torch

from focoos.ports import ModelOutput


@dataclass
class EoMTModelOutput(ModelOutput):
    masks: torch.Tensor  # [N, num_queries, H, W]
    logits: torch.Tensor  # [N, num_queries, num_classes]
    loss: Optional[dict]


@dataclass
class EoMTTargets:
    labels: torch.Tensor
    masks: torch.Tensor
