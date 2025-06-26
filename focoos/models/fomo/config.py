from dataclasses import dataclass, field
from typing import List

from focoos.nn.backbone.base import BackboneConfig
from focoos.ports import ModelConfig


@dataclass
class FOMOConfig(ModelConfig):
    
    backbone_config: BackboneConfig
    num_classes: int
    
    resolution: int = 640
    hidden_dim: int = 32
    activation: str = "relu"
    cut_point: str = "res3" # 1/8 input resolution
    reduction_factor: int = 8 # should match with cut_point
    freeze_backbone: bool = True
    loss_type: str = "bce_loss" # bce_loss, ce_loss, l1, l2, weighted_l1, weighted_l2 or any combination of them, concat them with "+"
    mask_threshold: float = 0.5
    
    pixel_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    pixel_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])
    