from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.fai_detr.config import DETRConfig
from focoos.models.fai_detr.ports import DETRModelOutput, DETRTargets
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import Boxes, ImageList, Instances
from focoos.utils.box import box_xyxy_to_cxcywh
from focoos.utils.logger import get_logger

logger = get_logger("DETRProcessor")


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances,
    output_height: int,
    output_width: int,
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    new_size = (output_height, output_width)
    output_width_tmp = output_width
    output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, boxes=results.boxes, scores=results.scores, classes=results.classes)

    assert results.boxes is not None, "Predictions must contain boxes!"
    results.boxes.scale(scale_x, scale_y)
    results.boxes.clip(results.image_size)

    results = results[results.boxes.nonempty()]  # type: ignore

    return results


class DETRProcessor(Processor):
    def __init__(self, config: DETRConfig):
        super().__init__(config)
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.resolution = config.resolution

    def preprocess(
        self,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
            list[DatasetEntry],
        ],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        image_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, list[DETRTargets]]:
        targets = []
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], DatasetEntry):
            images = [x.image.to(device) for x in inputs]  # type: ignore
            images = ImageList.from_tensors(
                tensors=images,
            )
            images_torch = images.tensor
            if self.training:
                # mask classification target
                gt_instances = [x.instances.to(device) for x in inputs]  # type: ignore
                h, w = images.tensor.shape[-2:]
                targets = []
                for targets_per_image in gt_instances:
                    assert targets_per_image.boxes is not None and targets_per_image.classes is not None, (
                        "boxes and classes are required for training"
                    )
                    image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=device)
                    gt_classes = targets_per_image.classes
                    gt_boxes = targets_per_image.boxes.tensor / image_size_xyxy
                    gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                    targets.append(DETRTargets(labels=gt_classes, boxes=gt_boxes))
        else:
            if self.training:
                raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore
            if image_size is not None:
                images_torch = torch.nn.functional.interpolate(
                    images_torch, size=(image_size, image_size), mode="bilinear", align_corners=False
                )
        return images_torch, targets

    def eval_postprocess(
        self,
        output: DETRModelOutput,
        batched_inputs: list[DatasetEntry],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Instances]]:
        top_k = top_k or self.top_k
        results = []
        box_cls, box_pred = output.logits, output.boxes
        batch_size = box_cls.shape[0]
        num_classes = box_cls.shape[-1]

        for i in range(batch_size):
            # Process results directly within the loop
            scores, labels, processed_box_pred = self._get_predictions(box_cls[i], box_pred[i], top_k, num_classes)

            boxes = Boxes(processed_box_pred)
            result = Instances(image_size=(1, 1), boxes=boxes, scores=scores, classes=labels)
            result = detector_postprocess(
                result, output_height=batched_inputs[i].height or 1, output_width=batched_inputs[i].width or 1
            )
            results.append({"instances": result})

        return results

    def _get_predictions(self, scores, boxes, top_k, num_classes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, index = torch.topk(scores.flatten(0), top_k, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        box_pred = boxes.gather(dim=0, index=index.unsqueeze(-1).repeat(1, boxes.shape[-1]))
        return scores, labels, box_pred

    def postprocess(
        self,
        output: DETRModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        class_names: list[str] = [],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[FocoosDetections]:
        # Extract image sizes from inputs

        top_k = top_k or self.top_k
        threshold = threshold or self.threshold

        image_sizes = self.get_image_sizes(inputs)

        results = []
        batch_size = output.boxes.shape[0]
        num_classes = output.logits.shape[-1]

        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )

        for i in range(batch_size):
            # Process results directly within the loop
            scores, labels, box_pred = self._get_predictions(output.logits[i], output.boxes[i], top_k, num_classes)

            # Apply threshold to filter out low-confidence predictions
            mask = scores > threshold
            box_pred = box_pred[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Multiply boxes by image size
            box_pred[:, 0::2] = box_pred[:, 0::2] * image_sizes[i][1]
            box_pred[:, 1::2] = box_pred[:, 1::2] * image_sizes[i][0]
            # Convert box coordinates to integers for pixel-precise bounding boxes
            box_pred = box_pred.round().to(torch.int32)

            # Convert tensor outputs to Python lists of floats
            py_box_pred = box_pred.detach().cpu().tolist()
            py_scores = scores.detach().cpu().tolist()
            py_labels = labels.detach().cpu().tolist()
            results.append(
                FocoosDetections(
                    detections=[
                        FocoosDet(
                            bbox=py_bp,
                            conf=py_s,
                            cls_id=py_l,
                            label=class_names[py_l] if class_names else None,
                        )
                        for py_bp, py_s, py_l in zip(py_box_pred, py_scores, py_labels)
                    ]
                )
            )

        return results

    def export_postprocess(
        self,
        output: Union[list[torch.Tensor], list[np.ndarray]],
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            list[np.ndarray],
            list[torch.Tensor],
        ],
        class_names: list[str] = [],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[FocoosDetections]:
        boxes = output[0]
        logits = output[1]
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        model_output = DETRModelOutput(boxes=boxes, logits=logits, loss=None)
        top_k = 300 if top_k is None else top_k
        threshold = 0.5 if threshold is None else threshold
        return self.postprocess(model_output, inputs, class_names, top_k, threshold)

    def get_dynamic_axes(self) -> DynamicAxes:
        return DynamicAxes(
            input_names=["images"],
            output_names=["boxes", "logits"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
