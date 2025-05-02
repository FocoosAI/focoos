from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.fai_mf.config import MaskFormerConfig
from focoos.models.fai_mf.ports import MaskFormerModelOutput, MaskFormerTargets
from focoos.models.fai_model import BaseProcessor
from focoos.ports import DatasetEntry, FocoosDet, FocoosDetections
from focoos.structures import BitMasks, ImageList, Instances
from focoos.utils.memory import retry_if_cuda_oom


class MaskFormerProcessor(BaseProcessor):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.config = config
        processing_functions = {
            "semantic": self.semantic_inference,
            "instance": self.instance_inference,
        }
        self.eval_output_name = "sem_seg" if config.postprocessing_type == "semantic" else "instances"
        assert config.postprocessing_type in processing_functions, (
            f"Invalid postprocessing type: {config.postprocessing_type}. Must be one of: {processing_functions.keys()}"
        )
        self.processing_fn = processing_functions[config.postprocessing_type]

        self.num_classes = config.num_classes
        self.top_k = config.top_k
        self.mask_threshold = config.mask_threshold

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
        training: bool,
        device: torch.device,
        dtype: torch.dtype,
        size_divisibility: int = 0,
        padding_constraints: Optional[Dict[str, int]] = None,
        resolution: Optional[int] = 640,
    ) -> tuple[torch.Tensor, list[MaskFormerTargets]]:
        targets = []
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], DatasetEntry):
            images = [x.image.to(device) for x in inputs]
            images = ImageList.from_tensors(
                tensors=images,
                # FIXME using size_divisibility in eval make detection break due to padding issue (in scaling bboxes)
                size_divisibility=size_divisibility if training or size_divisibility else 0,
                padding_constraints=padding_constraints,
            )
            images_torch = images.tensor
            if training:
                # mask classification target
                gt_instances = [x.instances.to(device) for x in inputs]
                h, w = images.tensor.shape[-2:]
                targets = []
                for targets_per_image in gt_instances:
                    gt_masks = targets_per_image.gt_masks
                    if len(gt_masks) > 0:
                        padded_masks = torch.zeros(
                            (gt_masks.shape[0], h, w),
                            dtype=gt_masks.dtype,
                            device=gt_masks.device,
                        )
                        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                    else:
                        padded_masks = gt_masks
                    cls_labels = targets_per_image.gt_classes
                    targets.append(MaskFormerTargets(labels=cls_labels, masks=padded_masks))
        else:
            if training:
                raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore
            if resolution is not None:
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

    def instance_inference(
        self,
        mask_cls,
        mask_pred,
    ) -> Instances:
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        num_queries = mask_pred.shape[0]

        # [Q, K]
        scores = mask_cls
        labels = (
            torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        )
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.top_k, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > self.mask_threshold).float()
        result.pred_boxes = BitMasks(mask_pred > self.mask_threshold).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
            result.pred_masks.flatten(1).sum(1) + 1e-6
        )
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def eval_postprocess(
        self,
        output: MaskFormerModelOutput,
        batched_inputs: list[DatasetEntry],
    ) -> list[dict[str, Union[Instances, torch.Tensor]]]:
        results = []
        cls_pred = output.logits
        mask_pred = output.masks

        for i in range(len(batched_inputs)):
            # get "augmented" images size and next original size
            size = batched_inputs[i].image.shape[-2:]  # type: ignore
            height = batched_inputs[i].height
            width = batched_inputs[i].width
            mask_pred_result = mask_pred[i]
            mask_cls_result = cls_pred[i]

            out_stride = size[1] // mask_pred_result.shape[2]
            mask_pred_result = mask_pred_result[:, : 1 + size[0] // out_stride, : 1 + size[1] // out_stride]

            def interpolate_image(image, size):
                return torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )[0]

            mask_pred_result = retry_if_cuda_oom(interpolate_image)(mask_pred_result, (height, width))
            result = self.processing_fn(mask_cls_result, mask_pred_result)
            results.append({self.eval_output_name: result})

        return results

    def postprocess(
        self,
        output: MaskFormerModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        top_k: int = 300,
        threshold: float = 0.5,
    ) -> list[FocoosDetections]:
        # Extract image sizes from inputs
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
            box_pred[:, 0::2] = box_pred[:, 0::2] * image_sizes[i][0]
            box_pred[:, 1::2] = box_pred[:, 1::2] * image_sizes[i][1]
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
                        )
                        for py_bp, py_s, py_l in zip(py_box_pred, py_scores, py_labels)
                    ]
                )
            )

        return results
