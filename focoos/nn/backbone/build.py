from focoos.nn.backbone.base import BackboneConfig


def load_backbone(config: BackboneConfig):
    # to avoid circular import
    from focoos.model_manager import AutoBackbone

    return AutoBackbone.from_config(config)
