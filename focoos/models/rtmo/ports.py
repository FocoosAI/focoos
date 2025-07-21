from dataclasses import dataclass
from typing import Literal

import torch

from focoos.models.yoloxpose.ports import KeypointOutput
from focoos.ports import ModelOutput

RTMOKeypointLossLiteral = Literal[
    "objectness", "boxes", "oks", "visibility", "classification", "bbox_aux", "mle", "classification_varifocal"
]
RTMOLoss = dict[RTMOKeypointLossLiteral, torch.Tensor]


@dataclass
class RTMOModelOutput(ModelOutput):
    outputs: KeypointOutput
    loss: RTMOLoss
