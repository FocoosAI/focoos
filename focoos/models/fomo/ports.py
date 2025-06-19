from dataclasses import dataclass
from typing import Optional

import torch

from focoos.ports import ModelOutput


@dataclass
class FOMOModelOutput(ModelOutput):
    logits: torch.Tensor
    loss: Optional[dict]


@dataclass
class FOMOTargets:
    labels: torch.Tensor
    boxes: torch.Tensor
    mask: torch.Tensor
