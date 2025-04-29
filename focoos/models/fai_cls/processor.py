from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetDict
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.ports import ClassificationTargets
from focoos.structures import ImageList


class ClassificationProcessor:
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
        if isinstance(inputs, (Image.Image, np.ndarray, torch.Tensor)):
            inputs_list = [inputs]
        else:
            inputs_list = inputs

        # Process each input based on its type
        processed_inputs = []
        for inp in inputs_list:
            # todo check for tensor of 4 dimesions.
            if isinstance(inp, Image.Image):
                inp = np.array(inp)
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)

            # Ensure input has correct shape and type
            if inp.dim() == 3:  # Add batch dimension if missing
                inp = inp.unsqueeze(0)
            if inp.shape[1] != 3 and inp.shape[-1] == 3:  # Convert HWC to CHW if needed
                inp = inp.permute(0, 3, 1, 2)

            processed_inputs.append(inp)

        # Stack all inputs into a single batch tensor
        # use pixel mean to get dtype -> If fp16, pixel_mean is fp16, so inputs will be fp16
        # TODO: this will break with different image sizes
        images_torch = torch.cat(processed_inputs, dim=0).to(device, dtype=dtype)
        if resolution is not None:
            images_torch = torch.nn.functional.interpolate(
                images_torch, size=resolution, mode="bilinear", align_corners=False
            )
        return images_torch, targets

    def eval_postprocess(self, logits: torch.Tensor, batch_inputs: List[Dict]) -> List[Dict]:
        """Post-process model outputs.

        Args:
            logits: Model output logits [N, num_classes]
            batch_inputs: Batch input metadata
        """
        if self.multi_label:
            probs = F.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
        results = []
        for probs_i in probs:
            results.append({"logits": probs_i})
        return results

    def postprocess(self, logits: torch.Tensor, batch_inputs: List[Dict]) -> List[Dict]:
        """Post-process model outputs.

        Args:
            logits: Model output logits [N, num_classes]
            batch_inputs: Batch input metadata

        Returns:
            List of processed results with class probabilities and predicted class
        """
        logits = logits.detach().cpu()
        probs = F.softmax(logits, dim=1)

        results = []
        for i, probs_i in enumerate(probs):
            top_prob, top_class = torch.max(probs_i, dim=0)

            result = {
                "probabilities": probs_i.numpy(),
                "predicted_class": top_class.item(),
                "confidence": top_prob.item(),
                "height": batch_inputs[i]["height"],
                "width": batch_inputs[i]["width"],
            }
            results.append(result)

        return results
