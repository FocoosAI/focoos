from dataclasses import dataclass, field
from typing import List, Literal

from focoos.nn.backbone.base import BackboneConfig
from focoos.ports import ModelConfig

PostprocessingType = Literal["semantic", "instance"]


@dataclass
class BisenetFormerConfig(ModelConfig):
    backbone_config: BackboneConfig
    num_classes: int

    num_queries: int = 100
    resolution: int = 640

    # Image detector configuration
    pixel_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    pixel_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])
    size_divisibility: int = 0

    # Sizing configuration
    pixel_decoder_out_dim: int = 256
    pixel_decoder_feat_dim: int = 256

    # Transformer decoder
    transformer_predictor_out_dim: int = 256
    transformer_predictor_hidden_dim: int = 256
    transformer_predictor_dec_layers: int = 6
    transformer_predictor_dim_feedforward: int = 1024
    # Head configuration
    head_out_dim: int = 256
    cls_sigmoid: bool = False

    # Inference configuration
    # Options: "semantic", "instance", "panoptic"
    postprocessing_type: PostprocessingType = "semantic"
    top_k: int = 300
    mask_threshold: float = 0.5
    predict_all_pixels: bool = False
    use_mask_score: bool = False
    filter_empty_masks: bool = False
    threshold: float = 0.5

    # Loss configuration
    criterion_deep_supervision: bool = True
    criterion_eos_coef: float = 0.1
    criterion_num_points: int = 12544

    weight_dict_loss_dice: int = 5
    weight_dict_loss_mask: int = 5
    weight_dict_loss_ce: int = 2

    matcher_cost_class: int = 2
    matcher_cost_mask: int = 5
    matcher_cost_dice: int = 5
