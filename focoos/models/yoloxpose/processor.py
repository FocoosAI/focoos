from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.yoloxpose.config import YOLOXPoseConfig
from focoos.models.yoloxpose.ports import KeypointOutput, KeypointTargets, YOLOXPoseModelOutput
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import Boxes, ImageList, Instances, Keypoints
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class YOLOXPoseProcessor(Processor):
    def __init__(self, config: YOLOXPoseConfig):
        super().__init__(config)
        self.nms_pre = config.nms_topk
        self.nms_thr = config.nms_thr
        self.score_thr = config.score_thr
        self.skeleton = config.skeleton
        self.flip_map = config.flip_map

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
        **kwargs,
    ) -> tuple[torch.Tensor, list[KeypointTargets]]:
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
                targets = []
                for targets_per_image in gt_instances:
                    assert targets_per_image.boxes is not None and targets_per_image.classes is not None, (
                        "boxes and classes are required for training"
                    )
                    assert targets_per_image.keypoints is not None and targets_per_image.areas is not None, (
                        "gt_keypoints and gt_areas are required for training"
                    )

                    gt_classes = targets_per_image.classes
                    gt_boxes = targets_per_image.boxes.tensor
                    gt_keypoints = targets_per_image.keypoints.tensor[..., :2]
                    gt_visibility = targets_per_image.keypoints.tensor[..., -1]
                    gt_visibility = torch.where(gt_visibility == 2, torch.tensor(1, device=device), gt_visibility)
                    gt_num_keypoints = torch.count_nonzero(
                        targets_per_image.keypoints.tensor.max(dim=2, keepdim=True).values, dim=1
                    ).squeeze(-1)
                    gt_areas = targets_per_image.areas
                    targets.append(
                        KeypointTargets(
                            labels=gt_classes,
                            bboxes=gt_boxes,
                            keypoints=gt_keypoints,
                            keypoints_visible=gt_visibility,
                            keypoints_visible_weights=gt_num_keypoints,
                            areas=gt_areas,
                            scores=None,
                            priors=None,
                        )
                    )
        else:
            if self.training:
                raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
            images_torch = self.get_tensors(inputs).to(device, dtype=dtype)  # type: ignore

        return images_torch, targets

    # TODO: implement nms threshold
    def postprocess(
        self,
        outputs: YOLOXPoseModelOutput,
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

        batch_size = outputs.outputs.scores.shape[0]

        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )

        for i in range(batch_size):
            results = FocoosDetections(detections=[])
            filter_mask = outputs.outputs.scores[i] > threshold
            if outputs.outputs.pred_bboxes is not None:
                output_boxes = outputs.outputs.pred_bboxes
            else:
                raise ValueError("Predictions must contain boxes!")

            size = image_sizes[i]
            h, w = int(size[0]), int(size[1])

            logger.debug(f"outputs.outputs.scores[i].shape: {outputs.outputs.scores[i].shape}")
            # Scores
            filtered_scores = outputs.outputs.scores[i][filter_mask].to("cpu").numpy().tolist()
            logger.debug(f"filtered_scores_shape: {outputs.outputs.scores[i][filter_mask].shape}")

            # Labels
            filtered_labels = outputs.outputs.labels[i][filter_mask].to("cpu").numpy().tolist()

            # Boxes
            filtered_boxes = outputs.outputs.pred_bboxes[i][filter_mask]
            logger.debug(f"filtered_boxes_shape: {filtered_boxes.shape}")
            output_boxes = filtered_boxes.clip(0, max(h, w))
            output_boxes = output_boxes.cpu().numpy().astype(int).tolist()

            # keypoints
            filtered_keypoints = outputs.outputs.pred_keypoints[i][filter_mask]
            keypoints_vis_expanded = outputs.outputs.keypoints_visible[i].unsqueeze(-1)
            keypoints_with_vis = torch.cat((filtered_keypoints, keypoints_vis_expanded), dim=2)
            keypoints = keypoints_with_vis.cpu().numpy().astype(int).tolist()

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
                            output_boxes, filtered_scores, filtered_labels, keypoints
                        )
                    ]
                )
            )
            res.append(results)

        return res

    def eval_postprocess(
        self, output: YOLOXPoseModelOutput, batched_inputs: list[DatasetEntry]
    ) -> list[dict[str, Instances]]:
        results = []
        batch_size = output.outputs.scores.shape[0]

        assert len(batched_inputs) == batch_size, (
            f"Expected batched_inputs {len(batched_inputs)} to match batch size {batch_size}"
        )

        for i in range(batch_size):
            # Get original image size
            original_height = batched_inputs[i].height or 1  # without augmentation
            original_width = batched_inputs[i].width or 1  # without augmentation
            image_size = (original_height, original_width)

            # Get predictions for this image
            scores = output.outputs.scores[i]
            labels = output.outputs.labels[i]
            pred_bboxes = output.outputs.pred_bboxes[i]
            pred_keypoints = output.outputs.pred_keypoints[i]
            keypoints_visible = output.outputs.keypoints_visible[i]

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

        # Create KeypointOutput and YOLOXPoseModelOutput
        keypoint_output = KeypointOutput(
            scores=scores.to(device),
            labels=labels.to(device),
            pred_bboxes=pred_bboxes.to(device),
            bbox_scores=bbox_scores.to(device),
            pred_keypoints=pred_keypoints.to(device),
            keypoint_scores=keypoint_scores.to(device),
            keypoints_visible=keypoints_visible.to(device),
        )

        model_output = YOLOXPoseModelOutput(outputs=keypoint_output, loss={})

        # Use the regular postprocess method
        threshold = threshold or self.score_thr
        # Filter inputs to exclude DatasetEntry type for postprocess
        filtered_inputs = [x for x in inputs if not isinstance(x, DatasetEntry)]
        return self.postprocess(model_output, filtered_inputs, threshold=threshold, **kwargs)

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
