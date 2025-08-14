from dataclasses import dataclass
from typing import Literal, Optional

import torch

from focoos.ports import ModelOutput

RTMOKeypointLossLiteral = Literal[
    "objectness", "boxes", "oks", "visibility", "classification", "bbox_aux", "mle", "classification_varifocal"
]
RTMOLoss = dict[RTMOKeypointLossLiteral, torch.Tensor]


@dataclass
class RTMOModelOutput(ModelOutput):
    scores: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    boxes_scores: torch.Tensor
    keypoints: torch.Tensor
    keypoints_scores: torch.Tensor
    keypoints_visible: torch.Tensor
    loss: Optional[RTMOLoss]
