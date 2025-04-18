from dataclasses import dataclass, field
from typing import List

from focoos.models.fai_model import ModelConfig


@dataclass
class RTDetrStdCConfig(ModelConfig):
    # Backbone configuration
    backbone_base: int = 64  # from json: "base": 64
    backbone_layers: List[int] = field(default_factory=lambda: [4, 5, 3])  # from json: "layers": [4, 5, 3]
    backbone_out_features: List[str] = field(default_factory=lambda: ["res2", "res3", "res4", "res5"])  # from json

    # Pixel decoder configuration
    pixel_decoder_out_dim: int = 128  # from json: "out_dim": 128
    pixel_decoder_feat_dim: int = 128  # from json: "feat_dim": 128
    pixel_decoder_dropout: float = 0.0  # from json
    pixel_decoder_nhead: int = 8  # from json
    pixel_decoder_expansion: float = 1.0  # from json
    pixel_decoder_dim_feedforward: int = 1024  # from json
    pixel_decoder_num_encoder_layers: int = 0  # from json

    # Head configuration
    head_in_channels: int = 128  # from json
    head_out_dim: int = 128  # from json
    num_classes: int = 80  # from json: COCO classes
    head_mask_on: bool = False  # from json
    head_cls_sigmoid: bool = True  # from json

    # Criterion configuration
    criterion_deep_supervision: bool = True  # from json
    criterion_eos_coef: float = 0.1  # from json
    criterion_losses: List[str] = field(default_factory=lambda: ["vfl", "boxes"])  # from json
    criterion_num_points: int = 0  # from json
    criterion_focal_alpha: float = 0.75  # from json
    criterion_focal_gamma: float = 2.0  # from json

    # Matcher configuration
    matcher_cost_class: int = 2  # from json
    matcher_cost_bbox: int = 5  # from json
    matcher_cost_giou: int = 2  # from json
    matcher_use_focal_loss: bool = True  # from json
    matcher_alpha: float = 0.25  # from json
    matcher_gamma: float = 2.0  # from json

    # Weight dictionary
    weight_dict_loss_vfl: int = 1  # from json
    weight_dict_loss_bbox: int = 5  # from json
    weight_dict_loss_giou: int = 2  # from json

    # Transformer predictor configuration
    transformer_predictor_in_channels: int = 128  # from json
    transformer_predictor_out_dim: int = 128  # from json
    transformer_predictor_num_classes: int = 80  # from json: COCO classes
    transformer_predictor_hidden_dim: int = 256  # from json
    transformer_predictor_mask_on: bool = False  # from json
    transformer_predictor_sigmoid: bool = True  # from json
    transformer_predictor_num_queries: int = 300  # from json
    transformer_predictor_nhead: int = 8  # from json
    transformer_predictor_dec_layers: int = 3  # from json: modified from 6 to 3 as in json
    transformer_predictor_dim_feedforward: int = 1024  # from json
    transformer_predictor_resolution: int = 640  # from json

    # Image detector configuration
    pixel_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])  # from json
    pixel_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])  # from json
    size_divisibility: int = -1  # from json: modified from 0 to -1
    ignore_value: int = 255  # from json
    mask_on: bool = False  # from json
