from typing import Union

import numpy as np
import torch
from PIL import Image

from focoos.models.fomo.config import FomoConfig
from focoos.models.fomo.ports import FOMOModelOutput, FOMOTargets
from focoos.ports import DatasetEntry, FocoosDetections
from focoos.processor.base_processor import Processor


class FOMOProcessor(Processor):
    def __init__(self, config: FomoConfig):
        super().__init__(config)

    def preprocess(self, inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
            list[DatasetEntry],
        ]) -> tuple[torch.Tensor, list[FOMOTargets]]:
        pass

    def postprocess(
        self, 
        outputs: FOMOModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
    ) -> list[FocoosDetections]:
        pass