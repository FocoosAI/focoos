from dataclasses import dataclass
from typing import Literal, Optional

import torch

from focoos.ports import DictClass, ModelOutput

KeypointLossLiteral = Literal[
    "objectness", "boxes", "oks", "visibility", "classification", "bbox_aux", "mle", "classification_varifocal"
]
KeypointLoss = dict[KeypointLossLiteral, torch.Tensor]


@dataclass
class KeypointTargets(DictClass):
    bboxes: Optional[torch.Tensor]
    scores: Optional[torch.Tensor]
    priors: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    keypoints: Optional[torch.Tensor]
    keypoints_visible: Optional[torch.Tensor]
    areas: Optional[torch.Tensor]


@dataclass
class OptSampleList(DictClass):
    boxes: torch.Tensor
    labels: torch.Tensor
    keypoints: torch.Tensor
    keypoints_vis: torch.Tensor
    areas: torch.Tensor
    gt_fields: torch.Tensor


@dataclass
class KeypointOutput(DictClass):
    scores: torch.Tensor
    labels: torch.Tensor
    pred_bboxes: torch.Tensor
    bbox_scores: torch.Tensor
    pred_keypoints: torch.Tensor
    keypoint_scores: torch.Tensor
    keypoints_visible: torch.Tensor


@dataclass
class RTMOModelOutput(ModelOutput):
    outputs: KeypointOutput
    loss: KeypointLoss
