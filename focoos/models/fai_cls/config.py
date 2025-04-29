from dataclasses import dataclass, field
from typing import List

from focoos.models.fai_model import ModelConfig
from focoos.nn.backbone.base import BackboneConfig


@dataclass
class ClassificationConfig(ModelConfig):
    backbone_config: BackboneConfig
    num_classes: int

    # Image classification configuration
    resolution: int = 224
    pixel_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    pixel_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])

    # Head configuration
    hidden_dim: int = 512
    dropout_rate: float = 0.2
    features: str = "res5"
    num_layers: int = 2

    # Loss configuration
    use_focal_loss: bool = False
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    multi_label: bool = False
