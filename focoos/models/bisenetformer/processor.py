import base64
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from focoos.models.bisenetformer.config import BisenetFormerConfig
from focoos.models.bisenetformer.ports import BisenetFormerOutput, BisenetFormerTargets
from focoos.models.fai_model import BaseProcessor
from focoos.ports import DatasetEntry, FocoosDet, FocoosDetections
from focoos.structures import BitMasks, ImageList, Instances
from focoos.utils.memory import retry_if_cuda_oom


def interpolate_image(image, size):
    return torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


def binary_mask_to_base64(binary_mask: np.ndarray) -> str:
    """
    Converts a binary mask (NumPy array) to a base64-encoded PNG image using OpenCV.

    This function takes a binary mask, where values of `True` represent the areas of interest (usually 1s)
    and `False` represents the background (usually 0s). The binary mask is then converted to an image,
    and this image is saved in PNG format and encoded into a base64 string.

    Args:
        binary_mask (np.ndarray): A 2D NumPy array with boolean values (`True`/`False`).

    Returns:
        str: A base64-encoded string representing the PNG image of the binary mask.
    """
    # Directly convert the binary mask to uint8 and multiply by 255 in one step
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Use OpenCV to encode the image as PNG
    success, encoded_image = cv2.imencode(".png", binary_mask)
    if not success:
        raise ValueError("Failed to encode image")

    # Encode the image to base64
    return base64.b64encode(encoded_image).decode("utf-8")


def masks_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """
    Converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Parameters:
        masks (np.ndarray): A 3D `np.array` of shape `(N, W, H)`
            containing 2D bool masks

    Returns:
        np.ndarray: A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask
    """
    # Vectorized approach to find bounding boxes
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    # Use np.any to quickly find rows and columns with True values
    for i, mask in enumerate(masks):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


class BisenetFormerProcessor(BaseProcessor):
    def __init__(self, config: BisenetFormerConfig):
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
    ) -> tuple[torch.Tensor, list[BisenetFormerTargets]]:
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
                    targets.append(BisenetFormerTargets(labels=cls_labels, masks=padded_masks))
        else:
            if training:
                raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore

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
        output: BisenetFormerOutput,
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

            mask_pred_result = retry_if_cuda_oom(interpolate_image)(mask_pred_result, (height, width))
            result = self.processing_fn(mask_cls_result, mask_pred_result)
            results.append({self.eval_output_name: result})

        return results

    def postprocess(
        self,
        output: BisenetFormerOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        class_names: list[str] = [],
        top_k: int = 300,
        threshold: float = 0.5,
        use_mask_score: bool = True,
        filter_empty_masks: bool = True,
        predict_all_pixels: bool = False,
    ) -> list[FocoosDetections]:
        # Extract image sizes from inputs
        image_sizes = self.get_image_sizes(inputs)
        batch_size = output.logits.shape[0]
        results = []
        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )

        cls_pred, mask_pred = (
            output.logits,
            output.masks,
        )  # B x Q; B x Q x H/out_stride x W/out_stride
        # softmax done before. # B x Q; B x Q
        scores, labels = cls_pred.max(-1)

        # # let's binarize the mask
        if predict_all_pixels:
            b, q, h, w = mask_pred.shape
            p = scores.view(b, q, 1, 1) * mask_pred
            out = p.argmax(dim=1)  # Shape: [b, h, w]

            # Initialize an empty tensor for bin_mask_pred
            bin_mask_pred = torch.zeros((b, q, h, w), dtype=torch.bool, device=mask_pred.device)

            # Process each batch instance separately
            for batch_idx in range(b):
                # Create a mask for each class in this batch
                for class_idx in range(q):
                    # Set True where the argmax equals this class index
                    bin_mask_pred[batch_idx, class_idx] = out[batch_idx] == class_idx

        else:
            bin_mask_pred = mask_pred >= self.mask_threshold  # B x Q x H x W

        if use_mask_score:
            bin_mask_pred = bin_mask_pred.int()
            # Quickfix to avoid num. instability.
            bin_mask_pred = bin_mask_pred * 1e-3
            mask_score = (bin_mask_pred * mask_pred).sum(-1).sum(-1) / (
                (bin_mask_pred).sum(-1).sum(-1) + 1e-5
            )  # add EPS to avoid division by 0
            # Multiply mask scores to class scores for final score
            scores = scores * mask_score  # B x Q

        if scores.shape[1] > top_k:
            scores, index = torch.topk(scores, top_k, dim=-1)
            labels = torch.gather(labels, dim=1, index=index)  # B x top_k_masks
            bin_mask_pred = torch.gather(
                bin_mask_pred,
                dim=1,
                index=index.unsqueeze(-1).unsqueeze(-1).tile(1, 1, *mask_pred.shape[-2:]),
            )  # B x top_k_masks x H x W

        # Filter based on the scores greather than threshold
        if threshold > 0:
            filter_mask = scores > threshold
            filter_mask = filter_mask.nonzero(as_tuple=True)
            scores = torch.gather(scores, dim=1, index=filter_mask[1].unsqueeze(0))
            labels = torch.gather(labels, dim=1, index=filter_mask[1].unsqueeze(0))
            bin_mask_pred = torch.gather(
                bin_mask_pred,
                dim=1,
                index=filter_mask[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *bin_mask_pred.shape[-2:]),
            )  # B x top_k_masks x H x W

        # Find masks with zero sum
        if filter_empty_masks:
            non_zero_masks = bin_mask_pred.sum(dim=(-2, -1)) > 1  # B x top_k_masks
            # Set scores and labels to 0 for empty masks
            # Get indices of non-zero masks
            non_zero_indices = (non_zero_masks).nonzero(as_tuple=True)
            # Filter scores, labels and bin_mask_pred to only keep non-zero masks
            scores = torch.gather(scores, dim=1, index=non_zero_indices[1].unsqueeze(0))
            labels = torch.gather(labels, dim=1, index=non_zero_indices[1].unsqueeze(0))
            bin_mask_pred = torch.gather(
                bin_mask_pred,
                dim=1,
                index=non_zero_indices[1]
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, *bin_mask_pred.shape[-2:]),
            )

        bin_mask_pred = bin_mask_pred.detach().cpu()
        scores = scores.detach().cpu()
        labels = labels.detach().cpu()

        for i in range(batch_size):
            if len(bin_mask_pred[i]) == 0:
                results.append(FocoosDetections(detections=[]))
                continue
            # interpolate mask pred to original size
            bin_mask_pred_resized = retry_if_cuda_oom(interpolate_image)(
                bin_mask_pred[i].float(), image_sizes[i]
            ).bool()

            if self.config.postprocessing_type == "instance":
                box_pred = masks_to_xyxy(bin_mask_pred_resized.numpy())
                py_box_pred = box_pred.tolist()
            else:
                py_box_pred = [None] * len(scores[i])

            py_scores = scores[i].tolist()
            py_labels = labels[i].tolist()
            py_mask_pred = bin_mask_pred_resized.numpy()

            results.append(
                FocoosDetections(
                    detections=[
                        FocoosDet(
                            bbox=py_bp,
                            conf=py_s,
                            cls_id=py_l,
                            mask=binary_mask_to_base64(py_mp),
                            label=class_names[py_l] if class_names else None,
                        )
                        for py_bp, py_s, py_l, py_mp in zip(py_box_pred, py_scores, py_labels, py_mask_pred)
                    ]
                )
            )

        return results
