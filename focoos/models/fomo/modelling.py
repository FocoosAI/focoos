import torch

from focoos.models.focoos_model import BaseModelNN
from focoos.models.fomo.config import FomoConfig
from focoos.models.fomo.ports import FOMOTargets
from focoos.nn.backbone.build import load_backbone


class Fomo(BaseModelNN):
    def __init__(self, config: FomoConfig):
        super().__init__(config)
        self._export = False
        self.config = config
        
        backbone = load_backbone(self.config.backbone_config)

    def forward(self, images: torch.Tensor, targets: list[FOMOTargets] = []):
        pass