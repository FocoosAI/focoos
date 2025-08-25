from dataclasses import dataclass, field
from typing import List, Literal, Tuple

from focoos.nn.backbone.base import BackboneConfig
from focoos.nn.backbone.csp_darknet import CSPConfig
from focoos.ports import ModelConfig

NormType = Literal["BN"]


@dataclass
class RTMOConfig(ModelConfig):
    backbone_config: BackboneConfig = field(default_factory=lambda: CSPConfig(size="small", use_pretrained=True))
    num_classes: int

    # encoder config
    transformer_embed_dims: int = 256
    transformer_num_heads: int = 8
    transformer_feedforward_channels: int = 1024
    transformer_dropout: float = 0.0
    transformer_encoder_layers: int = 1
    csp_layers: int = 1
    hidden_dim: int = 256
    output_dim: int = 256
    pe_temperature: int = 10000
    widen_factor: float = 0.5
    spe_learnable: bool = False
    output_indices: List[int] = field(default_factory=lambda: [1, 2])

    num_keypoints: int = 17
    in_channels: int = 256
    pose_vec_channels: int = 256
    cls_feat_channels: int = 256
    stacked_convs: int = 2
    featmap_strides: List[int] = field(default_factory=lambda: [16, 8])
    featmap_strides_pointgenerator: List[int] = field(default_factory=lambda: [16, 8])
    centralize_points_pointgenerator: bool = False

    overlaps_power: float = 0.5
    pixel_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pixel_std: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # DCC related params
    feat_channels_dcc: int = 128
    num_bins: Tuple[int, int] = (192, 256)
    spe_channels: int = 128
    gau_s: int = 128
    gau_expansion_factor: int = 2
    gau_dropout_rate: float = 0.0

    # processing config
    nms_topk: int = 1000
    nms_thr: float = 0.65
    score_thr: float = 0.1
    skeleton: list[tuple[int, int]] = field(default_factory=lambda: [])
    keypoints: list[str] = field(default_factory=lambda: [])
