from dataclasses import dataclass
from typing import Tuple

from focoos.ports import ModelConfig


@dataclass
class EoMTConfig(ModelConfig):
    # backbone_config: BackboneConfig

    num_classes: int
    num_queries: int
    im_size: Tuple[int, int] = (640, 640)
    num_blocks: int = 4
    masked_attn_enabled: bool = False

    # Processor-specific configuration
    postprocessing_type: str = "instance"  # "instance" or "semantic"
    mask_threshold: float = 0.5
    top_k: int = 100
    threshold: float = 0.5
    use_mask_score: bool = True
    predict_all_pixels: bool = False
