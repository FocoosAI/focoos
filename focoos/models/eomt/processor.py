from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.eomt.config import EoMTConfig
from focoos.models.eomt.ports import EoMTModelOutput, EoMTTargets
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import BitMasks, ImageList, Instances
from focoos.utils.memory import retry_if_cuda_oom
from focoos.utils.vision import binary_mask_to_base64, masks_to_xyxy, trim_mask


def interpolate_image(image, size):
    return torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


class EoMTProcessor(Processor):
    def __init__(self, config: EoMTConfig):
        super().__init__(config)

        # Processing functions for different segmentation types
        processing_functions = {
            "semantic": self.semantic_inference,
            "instance": self.instance_inference,
        }

        self.config = config
        self.postprocessing_type = config.postprocessing_type

        self.eval_output_name = "sem_seg" if config.postprocessing_type == "semantic" else "instances"

        assert config.postprocessing_type in processing_functions, (
            f"Invalid postprocessing type: {config.postprocessing_type}. Must be one of: {processing_functions.keys()}"
        )
        self.processing_fn = processing_functions[config.postprocessing_type]

        # Configuration parameters
        self.num_classes = config.num_classes
        self.num_queries = config.num_queries
        self.mask_threshold = config.mask_threshold
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.use_mask_score = config.use_mask_score
        self.predict_all_pixels = config.predict_all_pixels

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
    ) -> tuple[torch.Tensor, list[EoMTTargets]]:
        targets = []

        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], DatasetEntry):
            # Training mode with DatasetEntry inputs
            images = [x.image.to(device) for x in inputs]  # type: ignore
            images = ImageList.from_tensors(tensors=images)
            images_torch = images.tensor

            if self.training:
                # Create mask classification targets
                gt_instances = [x.instances.to(device) for x in inputs]  # type: ignore
                h, w = images.tensor.shape[-2:]
                targets = []

                for targets_per_image in gt_instances:
                    assert targets_per_image.masks is not None, "masks are required for training"
                    gt_masks = targets_per_image.masks.tensor

                    if len(gt_masks) > 0:
                        padded_masks = torch.zeros(
                            (gt_masks.shape[0], h, w),
                            dtype=gt_masks.dtype,
                            device=gt_masks.device,
                        )
                        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                    else:
                        padded_masks = gt_masks

                    assert targets_per_image.classes is not None, "classes are required for training"
                    cls_labels = targets_per_image.classes
                    targets.append(EoMTTargets(labels=cls_labels, masks=padded_masks))
        else:
            # Inference mode
            if self.training:
                raise ValueError("During training, inputs should be a list of DatasetEntry")
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore

        return images_torch, targets

    def semantic_inference(self, mask_cls, mask_pred) -> torch.Tensor:
        """Semantic segmentation inference."""
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def instance_inference(self, mask_cls, mask_pred) -> Instances:
        """Instance segmentation inference."""
        image_size = mask_pred.shape[-2:]
        num_queries = mask_pred.shape[0]

        # [Q, K] scores
        scores = mask_cls
        labels = (
            torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        )

        # Get top-k predictions
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.top_k, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]

        # Create binary masks and compute mask scores
        bin_masks = mask_pred > self.mask_threshold
        bin_masks = bin_masks * 1e-3
        mask_scores_per_image = (bin_masks.flatten(1) * mask_pred.flatten(1)).sum(1) / (
            bin_masks.flatten(1).sum(1) + 1e-6
        )

        # Create output instances
        masks = BitMasks(bin_masks.float())
        boxes = masks.get_bounding_boxes()
        scores = scores_per_image * mask_scores_per_image
        classes = labels_per_image

        return Instances(image_size, boxes=boxes, masks=masks, scores=scores, classes=classes)

    def eval_postprocess(
        self,
        output: Union[EoMTModelOutput, tuple[list[torch.Tensor], list[torch.Tensor]]],
        batched_inputs: list[DatasetEntry],
    ) -> list[dict[str, Union[Instances, torch.Tensor]]]:
        """Postprocess for evaluation."""
        results = []

        # Convert to standard format
        output = self._convert_raw_output_to_eomt_output(output)
        cls_pred = output.logits
        mask_pred = output.masks

        for i in range(len(batched_inputs)):
            # Get image sizes
            size = batched_inputs[i].image.shape[-2:]  # type: ignore
            height = batched_inputs[i].height
            width = batched_inputs[i].width

            mask_pred_result = mask_pred[i]
            mask_cls_result = cls_pred[i]

            # Resize mask predictions to original image size
            out_stride = size[1] // mask_pred_result.shape[2]
            mask_pred_result = mask_pred_result[:, : 1 + size[0] // out_stride, : 1 + size[1] // out_stride]
            mask_pred_result = retry_if_cuda_oom(interpolate_image)(mask_pred_result, (height, width))

            # Apply the appropriate processing function
            result = self.processing_fn(mask_cls_result, mask_pred_result)
            results.append({self.eval_output_name: result})

        return results

    def _convert_raw_output_to_eomt_output(
        self, output: Union[EoMTModelOutput, tuple[list[torch.Tensor], list[torch.Tensor]]]
    ) -> EoMTModelOutput:
        """Convert raw EOMT model output to EoMTModelOutput format."""
        if isinstance(output, tuple):
            mask_logits_per_layer, class_logits_per_layer = output
            # Use the final layer predictions
            masks = mask_logits_per_layer[-1]
            logits = class_logits_per_layer[-1]
            return EoMTModelOutput(masks=masks, logits=logits, loss=None)
        else:
            return output

    def postprocess(
        self,
        output: Union[EoMTModelOutput, tuple[list[torch.Tensor], list[torch.Tensor]]],
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
        use_mask_score: Optional[bool] = None,
        predict_all_pixels: Optional[bool] = None,
        **kwargs,
    ) -> list[FocoosDetections]:
        """Main postprocessing function for inference."""
        # Convert to standard format
        output = self._convert_raw_output_to_eomt_output(output)

        # Use provided parameters or fall back to defaults
        top_k = top_k or self.top_k
        threshold = threshold or self.threshold
        use_mask_score = use_mask_score if use_mask_score is not None else self.use_mask_score
        predict_all_pixels = predict_all_pixels if predict_all_pixels is not None else self.predict_all_pixels

        # Extract image sizes from inputs
        image_sizes = self.get_image_sizes(inputs)
        batch_size = output.logits.shape[0]
        results = []

        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )

        cls_pred, mask_pred = output.logits, output.masks
        # Apply softmax to get class probabilities (excluding background class)
        scores, labels = cls_pred.softmax(-1)[:, :, :-1].max(-1)

        # Binarize masks based on prediction type
        if predict_all_pixels:
            b, q, h, w = mask_pred.shape
            p = scores.view(b, q, 1, 1) * mask_pred
            out = p.argmax(dim=1)  # Shape: [b, h, w]

            # Initialize binary mask predictions
            bin_mask_pred = torch.zeros((b, q, h, w), dtype=torch.bool, device=mask_pred.device)
            for batch_idx in range(b):
                for class_idx in range(q):
                    bin_mask_pred[batch_idx, class_idx] = out[batch_idx] == class_idx
        else:
            bin_mask_pred = mask_pred >= self.mask_threshold

        # Filter out empty masks
        non_zero_masks = bin_mask_pred.sum(dim=(-2, -1)) > 1
        non_zero_indices = non_zero_masks.nonzero(as_tuple=True)

        # Keep only non-empty masks
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
        mask_pred = torch.gather(
            mask_pred,
            dim=1,
            index=non_zero_indices[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *mask_pred.shape[-2:]),
        )

        # Apply mask scoring if enabled
        if use_mask_score:
            bin_mask_pred_float = bin_mask_pred.float() * 1e-3
            mask_score = (bin_mask_pred_float * mask_pred).sum(-1).sum(-1) / (
                bin_mask_pred_float.sum(-1).sum(-1) + 1e-5
            )
            scores = scores * mask_score

        # Filter by threshold
        if threshold is not None and threshold > 0:
            filter_mask = scores > threshold
            filter_indices = filter_mask.nonzero(as_tuple=True)
            scores = torch.gather(scores, dim=1, index=filter_indices[1].unsqueeze(0))
            labels = torch.gather(labels, dim=1, index=filter_indices[1].unsqueeze(0))
            bin_mask_pred = torch.gather(
                bin_mask_pred,
                dim=1,
                index=filter_indices[1]
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, *bin_mask_pred.shape[-2:]),
            )

        # Move to CPU for final processing
        bin_mask_pred = bin_mask_pred.detach().cpu()
        scores = scores.detach().cpu()
        labels = labels.detach().cpu()

        # Process each image in the batch
        for i in range(batch_size):
            if len(bin_mask_pred[i]) == 0:
                results.append(FocoosDetections(detections=[]))
                continue

            # Resize masks to original image size
            bin_mask_pred_resized = retry_if_cuda_oom(interpolate_image)(
                bin_mask_pred[i].float(), image_sizes[i]
            ).bool()

            # Get bounding boxes from masks
            box_pred = masks_to_xyxy(bin_mask_pred_resized.numpy())
            py_box_pred = box_pred.tolist()
            py_scores = scores[i].tolist()
            py_labels = labels[i].tolist()
            py_mask_pred = bin_mask_pred_resized.numpy()

            # Create detection results
            results.append(
                FocoosDetections(
                    detections=[
                        FocoosDet(
                            bbox=py_bp,
                            conf=py_s,
                            cls_id=py_l,
                            mask=binary_mask_to_base64(trim_mask(py_mp, py_bp)),
                            label=class_names[py_l] if class_names and py_l < len(class_names) else None,
                        )
                        for py_bp, py_s, py_l, py_mp in zip(py_box_pred, py_scores, py_labels, py_mask_pred)
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
        threshold: Optional[float] = None,
        class_names: list[str] = [],
        **kwargs,
    ) -> list[FocoosDetections]:
        """Postprocess exported model outputs."""
        masks = output[0]
        logits = output[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)

        model_output = EoMTModelOutput(logits=logits.to(device), masks=masks.to(device), loss=None)
        return self.postprocess(
            model_output,
            inputs,
            class_names,
            threshold=threshold,
            **kwargs,
        )

    def get_dynamic_axes(self) -> DynamicAxes:
        """Get dynamic axes for model export."""
        return DynamicAxes(
            input_names=["images"],
            output_names=["masks", "logits"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
            },
        )
