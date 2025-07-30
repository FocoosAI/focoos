from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.rtmo.config import RTMOConfig
from focoos.models.rtmo.ports import RTMOModelOutput
from focoos.models.yoloxpose.ports import KeypointTargets
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import Boxes, ImageList, Instances, Keypoints
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class RTMOProcessor(Processor):
    def __init__(self, config: RTMOConfig, image_size: Optional[int] = None):
        super().__init__(config, image_size)

        self.score_thr = config.score_thr

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
        **kwargs,
    ) -> tuple[torch.Tensor, list[KeypointTargets]]:
        targets = []

        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], DatasetEntry):
            # Batch transfer images to device for better performance
            images = [x.image.to(device, non_blocking=True) for x in inputs]  # type: ignore
            images = ImageList.from_tensors(tensors=images)
            images_torch = images.tensor

            if self.training:
                # Batch transfer instances to device
                gt_instances = [x.instances.to(device) for x in inputs]  # type: ignore

                # Pre-create tensors for better performance
                one_tensor = torch.tensor(1, device=device, dtype=torch.long)

                for targets_per_image in gt_instances:
                    assert targets_per_image.boxes is not None and targets_per_image.classes is not None, (
                        "boxes and classes are required for training"
                    )
                    assert targets_per_image.keypoints is not None and targets_per_image.areas is not None, (
                        "gt_keypoints and gt_areas are required for training"
                    )

                    # Extract data efficiently
                    gt_classes = targets_per_image.classes
                    gt_boxes = targets_per_image.boxes.tensor
                    keypoints_tensor = targets_per_image.keypoints.tensor
                    gt_keypoints = keypoints_tensor[..., :2]
                    gt_visibility = keypoints_tensor[..., -1]

                    # Optimize visibility conversion - avoid repeated tensor creation
                    gt_visibility = torch.where(gt_visibility == 2, one_tensor, gt_visibility)
                    gt_areas = targets_per_image.areas

                    targets.append(
                        KeypointTargets(
                            labels=gt_classes,
                            bboxes=gt_boxes,
                            keypoints=gt_keypoints,
                            keypoints_visible=gt_visibility,
                            keypoints_visible_weights=None,
                            areas=gt_areas,
                            scores=None,
                            priors=None,
                        )
                    )
        else:
            if self.training:
                raise ValueError("During training, inputs should be a list of DatasetEntry")
            # Type cast is safe here since we know inputs is not list[DatasetEntry]
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype, non_blocking=True)  # type: ignore
            if self.image_size is not None:
                images_torch = torch.nn.functional.interpolate(
                    images_torch, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
                )
        return images_torch, targets

    def postprocess(
        self,
        outputs: RTMOModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        class_names: list[str] = [],
        threshold: float = 0.5,
        device="cuda:0",
        **kwargs,
    ) -> list[FocoosDetections]:
        threshold = threshold or self.score_thr

        image_sizes = self.get_image_sizes(inputs)

        res = []

        batch_size = outputs.scores.shape[0]

        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )

        for i in range(batch_size):
            filter_mask = outputs.scores[i] > threshold
            if outputs.pred_bboxes is not None:
                output_boxes = outputs.pred_bboxes
            else:
                raise ValueError("Predictions must contain boxes!")

            size = image_sizes[i]
            h, w = int(size[0]), int(size[1])
            scale = False
            if self.image_size is not None:
                scale_x = w / self.image_size
                scale_y = h / self.image_size
                scale = True

            # logger.debug(f"outputs.outputs.scores[i].shape: {outputs.outputs.scores[i].shape}")

            # Apply filtering to all outputs consistently
            filtered_scores = outputs.scores[i][filter_mask]
            filtered_labels = outputs.labels[i][filter_mask]
            filtered_boxes = outputs.pred_bboxes[i][filter_mask]

            # Handle keypoints with potential extra batch dimension
            pred_keypoints_i = outputs.pred_keypoints[i]
            if pred_keypoints_i.ndim == 4 and pred_keypoints_i.shape[0] == 1:
                pred_keypoints_i = pred_keypoints_i.squeeze(0)
            filtered_keypoints = pred_keypoints_i[filter_mask]

            keypoints_visible_i = outputs.keypoints_visible[i]
            if keypoints_visible_i.ndim == 3 and keypoints_visible_i.shape[0] == 1:
                keypoints_visible_i = keypoints_visible_i.squeeze(0)
            filtered_keypoints_visible = keypoints_visible_i[filter_mask]

            #  logger.debug(f"filtered_boxes.shape: {filtered_boxes.shape}")

            # Process keypoints with visibility
            keypoints_vis_expanded = filtered_keypoints_visible.unsqueeze(-1)
            keypoints_with_vis = torch.cat((filtered_keypoints, keypoints_vis_expanded), dim=2)

            if scale:
                # Multiply boxes by image size
                filtered_boxes[:, 0::2] *= scale_x
                filtered_boxes[:, 1::2] *= scale_y
                keypoints_with_vis[:, :, 0] *= scale_x
                keypoints_with_vis[:, :, 1] *= scale_y

            # Clip and map keypoints (x, y) as int
            keypoints_with_vis[:, :, 0] = keypoints_with_vis[:, :, 0].clip(0, w).int()
            keypoints_with_vis[:, :, 1] = keypoints_with_vis[:, :, 1].clip(0, h).int()

            # Process boxes
            output_boxes = filtered_boxes.clip(0, max(h, w))

            # Convert to CPU and numpy in one go for better performance
            output_boxes_np = output_boxes.cpu().numpy().astype(int).tolist()
            filtered_scores_np = filtered_scores.cpu().numpy().tolist()
            filtered_labels_np = filtered_labels.cpu().numpy().tolist()
            keypoints_np = keypoints_with_vis.cpu().numpy().tolist()

            res.append(
                FocoosDetections(
                    detections=[
                        FocoosDet(
                            bbox=box,
                            conf=score,
                            cls_id=label,
                            label=class_names[label] if class_names and label < len(class_names) else None,
                            keypoints=keypoint,
                        )
                        for box, score, label, keypoint in zip(
                            output_boxes_np, filtered_scores_np, filtered_labels_np, keypoints_np
                        )
                    ]
                )
            )

        return res

    def eval_postprocess(
        self, output: RTMOModelOutput, batched_inputs: list[DatasetEntry]
    ) -> list[dict[str, Instances]]:
        results = []
        batch_size = output.scores.shape[0]

        assert len(batched_inputs) == batch_size, (
            f"Expected batched_inputs {len(batched_inputs)} to match batch size {batch_size}"
        )

        for i in range(batch_size):
            # Get original image size
            original_height = batched_inputs[i].height or 1  # without augmentation
            original_width = batched_inputs[i].width or 1  # without augmentation
            image_size = (original_height, original_width)

            # Get predictions for this image
            scores = output.scores[i]
            labels = output.labels[i]
            pred_bboxes = output.pred_bboxes[i]
            pred_keypoints = output.pred_keypoints[i]
            keypoints_visible = output.keypoints_visible[i]

            # Scale predictions back to original image size
            if isinstance(batched_inputs[i].image, torch.Tensor):
                image_height, image_width = batched_inputs[i].image.shape[-2:]  # type: ignore
            else:
                image_height, image_width = original_height, original_width
            scale_x = original_width / image_width
            scale_y = original_height / image_height

            # Scale bounding boxes
            scaled_bboxes = pred_bboxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=pred_bboxes.device)
            scaled_bboxes = scaled_bboxes.clip(0, max(original_width, original_height))

            # Scale keypoints
            scaled_keypoints = pred_keypoints.clone()
            scaled_keypoints[:, :, 0] *= scale_x
            scaled_keypoints[:, :, 1] *= scale_y

            # Create keypoints with visibility
            keypoints_with_vis = torch.cat((scaled_keypoints, keypoints_visible.unsqueeze(-1)), dim=2)

            # Create Instances object
            boxes = Boxes(scaled_bboxes)
            keypoints = Keypoints(keypoints_with_vis)

            result = Instances(image_size=image_size, boxes=boxes, scores=scores, classes=labels, keypoints=keypoints)

            results.append({"instances": result})

        return results

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
            list[DatasetEntry],
        ],
        threshold: Optional[float] = None,
        **kwargs,
    ) -> list[FocoosDetections]:
        # The output from exported model should be a list of tensors in the order:
        # [scores, labels, pred_bboxes, bbox_scores, pred_keypoints, keypoint_scores, keypoints_visible]
        if len(output) != 7:
            raise ValueError(f"Expected 7 outputs from exported model, got {len(output)}")

        scores, labels, pred_bboxes, bbox_scores, pred_keypoints, keypoint_scores, keypoints_visible = output

        # Convert numpy arrays to tensors if needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(pred_bboxes, np.ndarray):
            pred_bboxes = torch.from_numpy(pred_bboxes)
        if isinstance(bbox_scores, np.ndarray):
            bbox_scores = torch.from_numpy(bbox_scores)
        if isinstance(pred_keypoints, np.ndarray):
            pred_keypoints = torch.from_numpy(pred_keypoints)
        if isinstance(keypoint_scores, np.ndarray):
            keypoint_scores = torch.from_numpy(keypoint_scores)
        if isinstance(keypoints_visible, np.ndarray):
            keypoints_visible = torch.from_numpy(keypoints_visible)

        model_output = RTMOModelOutput(
            scores=scores.to(device),
            labels=labels.to(device),
            pred_bboxes=pred_bboxes.to(device),
            bbox_scores=bbox_scores.to(device),
            pred_keypoints=pred_keypoints.to(device),
            keypoint_scores=keypoint_scores.to(device),
            keypoints_visible=keypoints_visible.to(device),
            loss=None,
        )

        # Use the regular postprocess method
        threshold = threshold or self.score_thr
        # Filter inputs to exclude DatasetEntry type for postprocess
        # Handle both single inputs and list inputs
        if isinstance(inputs, list):
            filtered_inputs = [x for x in inputs if not isinstance(x, DatasetEntry)]
        else:
            # Single input case - wrap in list if not DatasetEntry
            filtered_inputs = [inputs] if not isinstance(inputs, DatasetEntry) else []

        return self.postprocess(model_output, filtered_inputs, threshold=threshold, **kwargs)  # type: ignore

    def get_dynamic_axes(self) -> DynamicAxes:
        return DynamicAxes(
            input_names=["images"],
            output_names=[
                "scores",
                "labels",
                "pred_bboxes",
                "bbox_scores",
                "pred_keypoints",
                "keypoint_scores",
                "keypoints_visible",
            ],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "scores": {0: "batch"},
                "labels": {0: "batch"},
                "pred_bboxes": {0: "batch"},
                "bbox_scores": {0: "batch"},
                "pred_keypoints": {0: "batch"},
                "keypoint_scores": {0: "batch"},
                "keypoints_visible": {0: "batch"},
            },
        )
