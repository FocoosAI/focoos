from dataclasses import dataclass, field
from typing import List, Literal, Tuple

from focoos.nn.backbone.base import BackboneConfig
from focoos.ports import ModelConfig

NormType = Literal["BN"]


@dataclass
class YOLOXPoseConfig(ModelConfig):
    backbone_config: BackboneConfig
    num_classes: int

    num_keypoints: int = 17
    in_channels: int = 256
    pose_vec_channels: int = 256
    feat_channels: int = 256
    stacked_convs: int = 2
    activation: str = "relu"
    featmap_strides: List[int] = field(default_factory=lambda: [32, 16, 8])
    featmap_strides_pointgenerator: List[Tuple[int, int]] = field(default_factory=lambda: [(32, 32), (16, 16), (8, 8)])
    centralize_points_pointgenerator: bool = False
    bbox_padding: float = 1.25
    nms_pre: int = 1000
    nms_thr: float = 0.7
    score_thr: float = 0.01
    norm: NormType = "BN"
    use_aux_loss: bool = False
    overlaps_power: float = 1.0
    pixel_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    pixel_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])
    # yolo_neck
    neck_feat_dim: int = 256
    neck_out_dim: int = 256
    # TODO: In Anyma, depth (c2f_depth here) is a list [2, 4, 4, 2], but only the first element is used (see Anyma issue #160).
    c2f_depth: int = 2
