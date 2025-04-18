from dataclasses import dataclass, field
from typing import List

from focoos.models.fai_model import ModelConfig


@dataclass
class RTDetrResnetConfig(ModelConfig):
    # Backbone configuration
    backbone_depth: int = 50
    backbone_variant: str = "d"
    backbone_freeze_at: int = -1
    backbone_num_stages: int = 4
    backbone_freeze_norm: bool = False

    # Pixel decoder configuration
    pixel_decoder_out_dim: int = 256
    pixel_decoder_feat_dim: int = 256
    pixel_decoder_dropout: float = 0.0
    pixel_decoder_nhead: int = 8
    pixel_decoder_dim_feedforward: int = 1024
    pixel_decoder_num_encoder_layers: int = 1

    # Head configuration
    head_in_channels: int = 256
    head_out_dim: int = 256
    num_classes: int = 365
    head_mask_on: bool = False
    head_cls_sigmoid: bool = True

    # Criterion configuration
    criterion_deep_supervision: bool = True
    criterion_eos_coef: float = 0.1
    criterion_losses: List[str] = field(default_factory=lambda: ["vfl", "boxes"])
    criterion_num_points: int = 0
    criterion_focal_alpha: float = 0.75
    criterion_focal_gamma: float = 2.0

    # Matcher configuration
    matcher_cost_class: int = 2
    matcher_cost_bbox: int = 5
    matcher_cost_giou: int = 2
    matcher_use_focal_loss: bool = True
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0

    # Weight dictionary
    weight_dict_loss_vfl: int = 1
    weight_dict_loss_bbox: int = 5
    weight_dict_loss_giou: int = 2

    # Transformer predictor configuration
    transformer_predictor_in_channels: int = 256
    transformer_predictor_out_dim: int = 256
    transformer_predictor_num_classes: int = 365
    transformer_predictor_hidden_dim: int = 256
    transformer_predictor_mask_on: bool = False
    transformer_predictor_sigmoid: bool = True
    transformer_predictor_num_queries: int = 300
    transformer_predictor_nhead: int = 8
    transformer_predictor_dec_layers: int = 6
    transformer_predictor_dim_feedforward: int = 1024
    transformer_predictor_resolution: int = 640

    # Image detector configuration
    pixel_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    pixel_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])
    size_divisibility: int = 0
    ignore_value: int = 255
    mask_on: bool = False
