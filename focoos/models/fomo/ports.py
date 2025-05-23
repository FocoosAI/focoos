from dataclasses import dataclass
from typing import Optional

import torch

from focoos.ports import ModelOutput


@dataclass
class FOMOModelOutput(ModelOutput):
    boxes: torch.Tensor
    logits: torch.Tensor
    loss: Optional[dict]


@dataclass
class FOMOTargets:
    labels: torch.Tensor
    boxes: torch.Tensor
