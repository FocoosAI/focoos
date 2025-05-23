from focoos.nn.backbone.base import BackboneConfig
from focoos.ports import ModelConfig


class FomoConfig(ModelConfig):
    backbone_config = BackboneConfig