from dataclasses import dataclass
from typing import Optional

import torch

from focoos.ports import ModelOutput


@dataclass
class EoMTModelOutput(ModelOutput):
    masks: Optional[torch.Tensor] = None  # [N, num_queries, H, W]
    logits: Optional[torch.Tensor] = None  # [N, num_queries, num_classes]


@dataclass
class EoMTTargets:
    labels: torch.Tensor
    masks: torch.Tensor
