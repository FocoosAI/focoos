from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetDict
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.ports import ClassificationModelOutput, ClassificationTargets
from focoos.models.fai_model import BaseProcessor
from focoos.ports import DatasetEntry, FocoosDet, FocoosDetections
from focoos.structures import ImageList


class ClassificationProcessor(BaseProcessor):
    """Processor for image classification model inputs and outputs."""

    def __init__(self, config: ClassificationConfig):
        """Initialize the processor with model configuration.

        Args:
            config: Model configuration
        """
        self.config = config
        self.multi_label = config.multi_label

    def preprocess(
        self,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            List[Image.Image],
            List[np.ndarray],
            List[torch.Tensor],
            List[ClassificationDatasetDict],
        ],
        training: bool,
        device: torch.device,
        dtype: torch.dtype,
        resolution: Optional[int] = 640,
        size_divisibility: int = 0,
        padding_constraints: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, List[ClassificationTargets]]:
        """Process input images for model inference.

        Args:
            inputs: Input images in various formats
            training: Whether the model is in training mode
            device: Device to run the model on
            dtype: Data type to use for the model
            resolution: Resolution of the model

        Returns:
            Tuple of processed tensors and batch inputs metadata
        """
        targets = []
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], ClassificationDatasetDict):
            inputs: List[ClassificationDatasetDict]
            images = [x.image.to(device) for x in inputs]
            images = ImageList.from_tensors(
                tensors=images,
                size_divisibility=size_divisibility if training or size_divisibility else 0,
                padding_constraints=padding_constraints,
            )
            images_torch = images.tensor
            targets = [
                ClassificationTargets(labels=torch.tensor(x.label, dtype=torch.int64, device=device)) for x in inputs
            ]
            return images_torch, targets

        if training:
            raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
        images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore
        if resolution is not None:
            images_torch = torch.nn.functional.interpolate(
                images_torch, size=resolution, mode="bilinear", align_corners=False
            )
        return images_torch, targets

    def eval_postprocess(self, outputs: ClassificationModelOutput, inputs: list[DatasetEntry]) -> List[Dict]:
        """Post-process model outputs.

        Args:
            outputs: Model output
            inputs: Batch input metadata
        """
        if self.multi_label:
            probs = F.sigmoid(outputs.logits)
        else:
            probs = F.softmax(outputs.logits, dim=1)
        results = []
        for probs_i in probs:
            results.append({"logits": probs_i})
        return results

    def postprocess(
        self,
        outputs: ClassificationModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        class_names: list[str] = [],
    ) -> List[FocoosDetections]:
        """Post-process model outputs.

        Args:
            outputs: Model output
            inputs: Batch input metadata

        Returns:
            List of processed results with class probabilities and predicted class
        """
        logits = outputs.logits.detach().cpu()
        probs = F.softmax(logits, dim=1)

        results = []
        for i, probs_i in enumerate(probs):
            top_prob, top_class = torch.max(probs_i, dim=0)

            result = FocoosDetections(
                detections=[
                    FocoosDet(
                        conf=top_prob.item(),
                        cls_id=int(top_class.item()),
                        label=class_names[int(top_class.item())] if class_names else None,
                    )
                ]
            )
            results.append(result)

        return results
