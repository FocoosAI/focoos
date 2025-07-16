from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.models.yoloxpose.config import YOLOXPoseConfig
from focoos.models.yoloxpose.ports import KeypointTargets, YOLOXPoseModelOutput
from focoos.ports import DatasetEntry, DynamicAxes, FocoosDet, FocoosDetections
from focoos.processor.base_processor import Processor
from focoos.structures import ImageList
from focoos.utils.logger import get_logger
from focoos.utils.timer import took

logger = get_logger(__name__)


class YOLOXPoseProcessor(Processor):
    def __init__(self, config: YOLOXPoseConfig):
        super().__init__(config)
        self.nms_pre = config.nms_pre
        self.nms_thr = config.nms_thr
        self.score_thr = config.score_thr
        self.skeleton = config.skeleton
        self.flip_map = config.flip_map
        self.resolution = config.resolution

    @took
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
            if image_size is not None:
                images_torch = torch.nn.functional.interpolate(
                    images_torch, size=(image_size, image_size), mode="bilinear", align_corners=False
                )
        return images_torch, targets

    def _get_predictions(self, scores, boxes, top_k, num_classes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, index = torch.topk(scores.flatten(0), top_k, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        box_pred = boxes.gather(dim=0, index=index.unsqueeze(-1).repeat(1, boxes.shape[-1]))
        return scores, labels, box_pred

    # TODO: implement threshold
    @took
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

            scale_x = self.resolution / w
            scale_y = self.resolution / h
            logger.debug(f"scale_x: {scale_x}, scale_y: {scale_y}")
            logger.debug(f"outputs.outputs.scores[i].shape: {outputs.outputs.scores[i].shape}")
            # Scores
            filtered_scores = outputs.outputs.scores[i][filter_mask].to("cpu").numpy().tolist()
            logger.debug(f"filtered_scores_shape: {outputs.outputs.scores[i][filter_mask].shape}")

            # Labels
            filtered_labels = outputs.outputs.labels[i][filter_mask].to("cpu").numpy().tolist()

            # Boxes
            filtered_boxes = outputs.outputs.pred_bboxes[i][filter_mask]
            logger.debug(f"filtered_boxes_shape: {filtered_boxes.shape}")
            output_boxes = filtered_boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y]).to(device)
            output_boxes = output_boxes.clip(0, max(h, w))
            output_boxes = output_boxes.cpu().numpy().astype(int).tolist()

            # keypoints
            filtered_keypoints = outputs.outputs.pred_keypoints[i][filter_mask]
            keypoints_vis_expanded = outputs.outputs.keypoints_visible[i].unsqueeze(-1)
            keypoints_with_vis = torch.cat((filtered_keypoints, keypoints_vis_expanded), dim=2)
            keypoints_with_vis[:, :, 0] *= scale_x
            keypoints_with_vis[:, :, 1] *= scale_y
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
        pass

    def eval_postprocess(self, outputs: YOLOXPoseModelOutput, inputs: list[DatasetEntry]):
        pass

    def get_dynamic_axes(self) -> DynamicAxes:
        pass
