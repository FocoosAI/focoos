from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.fomo.config import FOMOConfig
from focoos.models.fomo.ports import FOMOModelOutput, FOMOTargets
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import Boxes, ImageList, Instances
from focoos.utils.box import box_xyxy_to_cxcywh


def interpolate_image(image, size):
    return torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]
    
    
def convert_boxes_to_masks(boxes, classes, image_size):
    height, width = image_size
    mask = -torch.ones((height, width), dtype=torch.int32, device=boxes.device)
    
    for box, class_idx in zip(boxes, classes):
        center_x, center_y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        mask[center_y, center_x] = class_idx
    
    return mask


class FOMOProcessor(Processor):
    def __init__(self, config: FOMOConfig):
        super().__init__(config)
        self.resolution = config.resolution
        self.mask_threshold = config.mask_threshold
        self.loss_type = config.loss_type

    def preprocess(self, inputs: Union[
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
            resolution: Optional[int] = None,
    ) -> tuple[torch.Tensor, list[FOMOTargets]]:
        targets = []
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], DatasetEntry):
            images = [x.image.to(device, dtype=dtype) for x in inputs]  # type: ignore
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
                    gt_boxes = targets_per_image.boxes.tensor
                    gt_mask = convert_boxes_to_masks(gt_boxes, gt_classes, (h, w))
                    gt_boxes = gt_boxes / image_size_xyxy
                    gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                    targets.append(FOMOTargets(labels=gt_classes, boxes=gt_boxes, mask=gt_mask))
        else:
            if self.training:
                raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore
            resolution = resolution or self.resolution

            images_torch = torch.nn.functional.interpolate(
                images_torch, size=resolution, mode="bilinear", align_corners=False
            )
            # Normalize the inputs
        return images_torch, targets
    
    def semantic_inference(
        self,
        mask_cls,
        mask_pred,
    ) -> torch.Tensor:
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
    '''
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
        class_names: list[str] = [],
        top_k: Optional[int] = None,
        threshold: Optional[float] = 0.5,
    ) -> list[FocoosDetections]:
        
        OFFSET = 1 # offset to turn center of the box to box corners
        scores = outputs.logits.sigmoid()
        batch_size = scores.shape[0]
        scores_sizes = scores.shape[-2:]
        image_sizes = self.get_image_sizes(inputs)
        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )
        batch_idxs, class_ids, y_idxs, x_idxs = [x.tolist() for x in torch.where(scores > threshold)]
        
        results = []
        for batch in range(batch_size):
            detections = []
            for batch_idx, class_id, y_idx, x_idx in zip(batch_idxs, class_ids, y_idxs, x_idxs):
                score = scores[batch_idx, class_id, y_idx, x_idx].item()
                x_idx = round(x_idx * image_sizes[batch][1] / scores_sizes[1])
                y_idx = round(y_idx * image_sizes[batch][0] / scores_sizes[0])
                bbox = [x_idx-OFFSET, y_idx-OFFSET, x_idx+OFFSET, y_idx+OFFSET]
                detections.append(FocoosDet(bbox=bbox, conf=score, cls_id=class_id, label=class_names[class_id]))
            results.append(FocoosDetections(detections=detections))
                
        return results
    '''

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
        class_names: list[str] = [],
        top_k: Optional[int] = None,
        threshold: Optional[float] = 0.5,
    ):
                
        return outputs.logits
    
    def eval_postprocess(
        self,
        output: FOMOModelOutput,
        batched_inputs: list[DatasetEntry],
    ) -> list[dict[str, Union[Instances, torch.Tensor]]]:
        
        results = []
        mask_pred = (output.logits.sigmoid()).to(torch.float32)
        
        if self.loss_type == "bce_loss":
            for i in range(len(batched_inputs)):
                mask_height, mask_width = tuple(mask_pred[i].shape[-2:])
                background_mask = torch.zeros((mask_height, mask_width), dtype=torch.float32, device=mask_pred.device).unsqueeze(0).unsqueeze(0) + self.mask_threshold
                mask_pred_result = torch.cat([background_mask, mask_pred], dim=1).squeeze(0)
                results.append({'instances': mask_pred_result})
        elif self.loss_type == "ce_loss":
            for i in range(len(batched_inputs)):
                results.append({'instances': mask_pred[i]})
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        
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
    ) -> list[FocoosDetections]:
        pass
    
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