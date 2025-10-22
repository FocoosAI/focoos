# Copyright (c) FocoosAI
"""
Detection evaluator using torchmetrics for Lightning training.

This evaluator uses torchmetrics.detection.MeanAveragePrecision which is:
- Faster than pycocotools
- Better integrated with PyTorch Lightning
- More memory efficient
- GPU accelerated
"""

from collections import OrderedDict
from typing import List, Literal

import torch
from torchmetrics.detection import MeanAveragePrecision

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import DatasetEntry
from focoos.utils.logger import get_logger

from .evaluator import DatasetEvaluator

logger = get_logger("DetectionLightningEvaluator")


class DetectionLightningEvaluator(DatasetEvaluator):
    """
    Evaluate object detection and instance segmentation predictions using torchmetrics.

    This evaluator provides a drop-in replacement for DetectionEvaluator but uses
    torchmetrics for faster and more efficient metric computation.

    Args:
        dataset_dict: DictDataset containing the validation data
        task: Type of detection task - "bbox" for detection, "segm" for instance segmentation
        distributed: Whether running in distributed mode (handled by Lightning)
        device: Device to run metrics computation on

    Example:
        >>> evaluator = DetectionLightningEvaluator(
        ...     dataset_dict=val_dataset,
        ...     task="bbox",
        ...     distributed=False,
        ... )
        >>> evaluator.reset()
        >>> evaluator.process(inputs, predictions)
        >>> metrics = evaluator.evaluate()
    """

    def __init__(
        self,
        dataset_dict: DictDataset,
        task: Literal["bbox", "segm"] = "bbox",
        distributed: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(dataset_dict)
        self.iou_type: Literal["bbox", "segm"] = task  # "bbox" or "segm"
        self._distributed = distributed
        self.device = device

        # Initialize torchmetrics MeanAveragePrecision
        self.metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type=self.iou_type,
            class_metrics=True,
        ).to(device)

        logger.info(f"Initialized DetectionLightningEvaluator with task={task}")

    def reset(self):
        """Reset the metric for a new evaluation round."""
        self.metric.reset()

    def process(self, inputs: List[DatasetEntry], outputs: List[dict]):
        """
        Process predictions and ground truth.

        Args:
            inputs: List of DatasetEntry containing ground truth
            outputs: List of predictions from the model
        """
        if not outputs or len(outputs) == 0:
            logger.warning("No outputs to process")
            return

        preds = []
        targets = []

        for inp, pred in zip(inputs, outputs):
            # Convert prediction to torchmetrics format
            if "instances" in pred and pred["instances"] is not None:
                instances = pred["instances"]
                if len(instances) > 0:
                    pred_dict = {
                        "boxes": instances.boxes.tensor.to(self.device),
                        "scores": instances.scores.to(self.device),
                        "labels": instances.classes.to(self.device),  # Use 'classes' not 'labels'
                    }
                    preds.append(pred_dict)
                else:
                    # Empty prediction
                    preds.append(
                        {
                            "boxes": torch.empty((0, 4), device=self.device),
                            "scores": torch.empty((0,), device=self.device),
                            "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                        }
                    )
            else:
                # Empty prediction
                preds.append(
                    {
                        "boxes": torch.empty((0, 4), device=self.device),
                        "scores": torch.empty((0,), device=self.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                    }
                )

            # Convert ground truth to torchmetrics format
            if inp.instances is not None and len(inp.instances) > 0:
                # Check that boxes and classes exist
                if inp.instances.boxes is not None and inp.instances.classes is not None:
                    target_dict = {
                        "boxes": inp.instances.boxes.tensor.to(self.device),
                        "labels": inp.instances.classes.to(self.device),  # Use 'classes' not 'labels'
                    }
                    targets.append(target_dict)
                else:
                    # Missing annotations
                    targets.append(
                        {
                            "boxes": torch.empty((0, 4), device=self.device),
                            "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                        }
                    )
            else:
                # Empty ground truth
                targets.append(
                    {
                        "boxes": torch.empty((0, 4), device=self.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=self.device),
                    }
                )

        # Update metric
        self.metric.update(preds, targets)

    def evaluate(self) -> dict:
        """
        Compute and return evaluation metrics.

        Returns:
            dict: Dictionary with task as key and metrics dict as value
                Format: {task_name: {"AP": XX.XX, "AP50": XX.XX, ..., "per_category_ap": {...}}}
        """
        # Compute metrics
        metrics_dict = self.metric.compute()

        # Convert torchmetrics format to COCO format
        # torchmetrics returns values in [0, 1], COCO format uses [0, 100]
        metric_mapping = {
            "map": "AP",  # mAP @ IoU=0.50:0.95
            "map_50": "AP50",  # mAP @ IoU=0.50
            "map_75": "AP75",  # mAP @ IoU=0.75
            "map_small": "APs",  # mAP for small objects
            "map_medium": "APm",  # mAP for medium objects
            "map_large": "APl",  # mAP for large objects
            "mar_1": "AR@1",  # AR with 1 detection per image
            "mar_10": "AR@10",  # AR with 10 detections per image
            "mar_100": "AR@100",  # AR with 100 detections per image
            "mar_small": "ARs",  # AR for small objects
            "mar_medium": "ARm",  # AR for medium objects
            "mar_large": "ARl",  # AR for large objects
        }

        results = OrderedDict()
        task_results = {}

        for torchmetric_name, coco_name in metric_mapping.items():
            if torchmetric_name in metrics_dict:
                metric_value = metrics_dict[torchmetric_name]

                # Convert to float and scale to [0, 100]
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()

                metric_value = float(metric_value) * 100.0

                # Only add if not NaN
                if not torch.isnan(torch.tensor(metric_value)):
                    task_results[coco_name] = metric_value

        # Add per-category AP if available (compatible with old DetectionEvaluator format)
        if "map_per_class" in metrics_dict:
            per_class_ap = metrics_dict["map_per_class"]
            if isinstance(per_class_ap, torch.Tensor):
                # Convert to [0, 100] scale
                per_class_ap = per_class_ap * 100.0

                # Get class names from dataset metadata (same as old DetectionEvaluator)
                class_names = None
                if hasattr(self.dataset_dict, "metadata") and hasattr(self.dataset_dict.metadata, "thing_classes"):
                    class_names = self.dataset_dict.metadata.thing_classes

                # Add per-category AP using the same format as old DetectionEvaluator: "AP-{class_name}"
                for class_idx, ap_value in enumerate(per_class_ap):
                    if not torch.isnan(ap_value):
                        class_name = (
                            class_names[class_idx]
                            if class_names and class_idx < len(class_names)
                            else f"class_{class_idx}"
                        )
                        # Use "AP-{class_name}" format for compatibility
                        task_results[f"AP-{class_name}"] = ap_value.item()

        results[self.iou_type] = task_results

        logger.info(f"Evaluation completed with {len(task_results)} metrics")
        logger.info(f"Metrics: {[(k, v) for k, v in task_results.items()]}")
        return results
