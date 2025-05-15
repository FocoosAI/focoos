from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetDict
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.ports import ClassificationModelOutput, ClassificationTargets
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import ImageList


class ClassificationProcessor(Processor):
    """Processor for image classification model inputs and outputs."""

    def __init__(self, config: ClassificationConfig):
        """Initialize the processor with model configuration.

        Args:
            config: Model configuration
        """
        super().__init__(config)
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
        device: torch.device,
        dtype: torch.dtype,
        resolution: Optional[int] = 640,
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
            class_data_dict: List[ClassificationDatasetDict] = inputs  # type: ignore
            images = [x.image.to(device) for x in class_data_dict]  # type: ignore
            images = ImageList.from_tensors(
                tensors=images,
            )
            images_torch = images.tensor
            targets = [
                ClassificationTargets(labels=torch.tensor(x.label, dtype=torch.int64, device=device))
                for x in class_data_dict  # type: ignore
            ]
            return images_torch, targets

        if self.training:
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

    def tensors_to_model_output(
        self, tensors: Union[list[np.ndarray], list[torch.Tensor]]
    ) -> ClassificationModelOutput:
        """
        Convert a list of tensors or numpy arrays to a ClassificationModelOutput.

        Args:
            tensors: List of tensors or numpy arrays

        Returns:
            ClassificationModelOutput
        """
        if not (isinstance(tensors, (list, tuple)) and len(tensors) == 1):
            raise ValueError(
                f"Expected a list or tuple of 1 element, got {type(tensors)} with length {len(tensors) if hasattr(tensors, '__len__') else 'N/A'}"
            )
        if isinstance(tensors[0], np.ndarray):
            new_tensor = torch.from_numpy(tensors[0])
        else:
            new_tensor = tensors[0]
        return ClassificationModelOutput(logits=new_tensor, loss=None)

    def get_dynamic_axes(self) -> DynamicAxes:
        return DynamicAxes(
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch"},
            },
        )

    def export_postprocess(
        self,
        output: Union[list[torch.Tensor], list[np.ndarray]],
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        **kwargs,
    ) -> list[FocoosDetections]:
        logits = output[0]
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        model_output = ClassificationModelOutput(logits=logits, loss=None)
        return self.postprocess(model_output, inputs, **kwargs)
