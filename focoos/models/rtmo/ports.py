from dataclasses import dataclass
from typing import Literal, Optional

import torch

from focoos.ports import DictClass, ModelOutput

RTMOKeypointLossLiteral = Literal[
    "objectness", "boxes", "oks", "visibility", "classification", "bbox_aux", "mle", "classification_varifocal"
]
RTMOLoss = dict[RTMOKeypointLossLiteral, torch.Tensor]


@dataclass
class KeypointTargets(DictClass):
    boxes: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    priors: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    keypoints: Optional[torch.Tensor]
    keypoints_visible: Optional[torch.Tensor]
    keypoints_visible_weights: Optional[torch.Tensor]
    areas: Optional[torch.Tensor]


@dataclass
class KeypointOutput(DictClass):
    scores: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    boxes_scores: torch.Tensor
    keypoints: torch.Tensor
    keypoints_scores: torch.Tensor
    keypoints_visible: torch.Tensor


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
