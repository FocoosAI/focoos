from dataclasses import dataclass
from typing import Optional

import torch

from focoos.ports import ModelOutput


@dataclass
class ClassificationModelOutput(ModelOutput):
    logits: torch.Tensor  # [N, num_classes]
    loss: Optional[dict]


@dataclass
class ClassificationTargets:
    labels: torch.Tensor  # [N], class indices
