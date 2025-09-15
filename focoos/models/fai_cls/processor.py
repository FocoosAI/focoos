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

    def __init__(self, config: ClassificationConfig, image_size: Optional[int] = None):
        """Initialize the processor with model configuration.

        Args:
            config: Model configuration
        """
        super().__init__(config, image_size)
        self.config = config
        self.pixel_mean = torch.Tensor(config.pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(config.pixel_std).view(-1, 1, 1)
        self.num_classes = config.num_classes

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
        dtype: torch.dtype = torch.float32,
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

            labels = torch.zeros(len(class_data_dict), self.num_classes, dtype=torch.int, device=device)
            for i, x in enumerate(class_data_dict):
                if x.label is not None:
                    labels[i, x.label] = 1
            targets = [ClassificationTargets(labels=labels[i]) for i in range(len(class_data_dict))]

            return images_torch, targets

        if self.training:
            raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
        target_size = (self.image_size, self.image_size) if self.image_size is not None else None
        if target_size is not None:
            print("Resizing to", target_size)
        images_torch = self.get_torch_batch(
            inputs,  # type: ignore
            target_size=target_size,
            device=device,
            dtype=dtype,
        )
        # self.pixel_mean, self.pixel_std = self.pixel_mean.to(device), self.pixel_std.to(device)
        # images_torch = (images_torch - self.pixel_mean) / self.pixel_std  # type: ignore
        return images_torch, targets

    def eval_postprocess(self, outputs: ClassificationModelOutput, inputs: list[DatasetEntry]) -> List[Dict]:
        """Post-process model outputs.

        Args:
            outputs: Model output
            inputs: Batch input metadata
        """
        probs = F.sigmoid(outputs.logits)

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
        threshold: Optional[float] = None,
    ) -> List[FocoosDetections]:
        """Post-process model outputs.

        Args:
            outputs: Model output
            inputs: Batch input metadata

        Returns:
            List of processed results with class probabilities and predicted classes
        """
        logits = outputs.logits.detach().cpu()

        probs = F.sigmoid(logits)
        threshold = threshold or 0.5

        results = []
        for i, probs_i in enumerate(probs):
            # For multi-label, return all classes above threshold
            predicted_classes = (probs_i > threshold).nonzero(as_tuple=True)[0]
            detections = []
            for cls_id in predicted_classes:
                detections.append(
                    FocoosDet(
                        conf=probs_i[cls_id].item(),
                        cls_id=int(cls_id.item()),
                        label=class_names[int(cls_id.item())]
                        if class_names and int(cls_id.item()) < len(class_names)
                        else None,
                    )
                )

            result = FocoosDetections(detections=detections)
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
        threshold: Optional[float] = None,
        **kwargs,
    ) -> list[FocoosDetections]:
        logits = output[0]
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        model_output = ClassificationModelOutput(logits=logits, loss=None)
        return self.postprocess(model_output, inputs, threshold=threshold, **kwargs)
