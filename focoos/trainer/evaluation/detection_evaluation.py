# Copyright (c) FocoosAI
import copy
import itertools
from collections import OrderedDict
from typing import List

import numpy as np
import pycocotools.mask as mask_util
import torch

from focoos.data.mappers.detection_dataset_mapper import DetectionDatasetDict

try:
    import faster_coco_eval

    # Replace pycocotools with faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import focoos.utils.distributed.comm as comm
from focoos.data.datasets.dict_dataset import DictDataset
from focoos.structures import BoxMode
from focoos.utils.logger import create_small_table, get_logger

from .evaluator import DatasetEvaluator

logger = get_logger("evaluation")


class DetectionEvaluator(DatasetEvaluator):
    """
    Evaluate object detection and instance segmentation predictions using COCO-style metrics.

    This evaluator supports evaluating:
    - Bounding box detection (task="bbox")
    - Instance segmentation (task="segm")

    The metrics include:
    - Average Precision (AP) at different IoU thresholds
    - AP for different object scales (small, medium, large)
    - Per-category AP

    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    This evaluator can be used with any dataset that follows the COCO data format.
    """

    def __init__(
        self,
        dataset_dict: DictDataset,
        task="bbox",
        distributed=True,
    ):
        """
        Args:
            dataset_dict: Dataset in DictDataset format containing the ground truth annotations
            task: Evaluation task, one of "bbox", "segm", or "keypoints"
            distributed: If True, evaluation will be distributed across multiple processes.
                       If False, evaluation runs only in the current process.
        """
        self._distributed = distributed
        self.dataset_dict = dataset_dict
        self.metadata = self.dataset_dict.metadata

        assert task in {"bbox", "segm"}, f"Got unknown task: {task}!"
        self.iou_type = task

        self.num_classes = self.metadata.num_classes
        self.class_names = self.metadata.thing_classes

        self.cpu_device = torch.device("cpu")

        self._predictions = []
        # self._inputs = []

    def reset(self):
        """Clear stored predictions and inputs."""
        self._predictions = []
        # self._inputs = []

    def process(self, inputs: List[DetectionDatasetDict], outputs):
        """
        Process one batch of model inputs and outputs.

        Args:
            inputs: List of dicts containing input image metadata like "image_id", "height", "width"
            outputs: List of dicts containing model predictions with key "instances" containing
                    detection/segmentation results as Instances objects
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input.image_id}

            if "instances" in output:
                # in the dataset mapper, we did not applied augmentations, so we can directly use the gt instances
                prediction["instances"] = self.instances_to_coco_json(
                    output["instances"].to(self.cpu_device), input.image_id
                )
                self._predictions.append(prediction)
            else:
                raise Exception("No instances in output?!")

    def evaluate(self):
        """
        Evaluate all stored predictions against ground truth.

        For distributed training, aggregates predictions from all workers on rank 0.

        Returns:
            dict: Evaluation results with metrics like AP, AP50, AP75 etc.
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        logger.info("Preparing results for COCO format ...")
        predictions = list(
            itertools.chain(*[x["instances"] for x in predictions])
        )  # this is a list of dicts (see the conversion below)

        inputs = []
        images = {}
        logger.info(f"predictions: {len(predictions)}")
        for x in predictions:
            if x["image_id"] not in images:
                in_ = self.dataset_dict[x["image_id"]]
                for ann in in_["annotations"]:
                    ann["image_id"] = x["image_id"]
                    inputs.append(ann)
                in_.pop("annotations")
                in_["id"] = x["image_id"]
                images[x["image_id"]] = in_
        logger.info(f"inputs: {len(inputs)}")

        self._results = OrderedDict()
        if len(predictions) > 0:
            self._results[self.iou_type] = self._eval_predictions(predictions, inputs, images)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions, inputs, images):
        """
        Evaluate predictions using COCO API.

        Args:
            predictions: List of dicts containing model predictions
            inputs: List of dicts containing ground truth annotations
        """
        coco_inputs = inputs
        coco_results = predictions

        categories = [
            {
                "id": idx,
                "name": x,
            }
            for idx, x in enumerate(self.dataset_dict.metadata.thing_classes)
        ]

        coco_eval = (
            self._evaluate_predictions_on_coco(
                images,
                categories,
                coco_inputs,
                coco_results,
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )

        res = self._derive_coco_results(coco_eval, class_names=self.metadata.thing_classes)
        return res

    def _evaluate_predictions_on_coco(
        self,
        images,
        categories,
        coco_gt,
        coco_results,
    ):
        """
        Evaluate predictions using COCO evaluation API.

        Args:
            images: List of dicts with image metadata
            categories: List of dicts with category information
            coco_gt: List of ground truth annotations in COCO format
            coco_results: List of predictions in COCO format

        Returns:
            COCOeval object with evaluation results
        """
        assert len(coco_results) > 0
        # Basically, we have all the ingredients to do this. Before let's try with COCOeval
        coco_gt = create_coco(images, categories, coco_gt, self.iou_type)
        coco_dt = create_coco(images, categories, coco_results, self.iou_type)

        coco_eval = COCOeval(coco_gt, coco_dt, self.iou_type)
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    def _derive_coco_results(self, coco_eval, class_names=None):
        """
        Derive evaluation metrics from COCOeval results.

        Args:
            coco_eval: COCOeval object containing evaluation results
            class_names: List of category names for per-category metrics

        Returns:
            dict: Results including:
                - Overall metrics (AP, AP50, AP75, APs, APm, APl)
                - Per-category AP if class_names provided
        """
        iou_type = self.iou_type
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
        if not np.isfinite(sum(results.values())):
            logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        logger.info(f"precisions: {precisions.shape} class_names: {class_names}")

        assert len(class_names) == precisions.shape[2], (
            f"Found {len(class_names)} classes, but precision has dimension {precisions.shape[2]}"
        )

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def instances_to_coco_json(self, instances, img_id):
        """
        Convert Instances predictions to COCO json format.

        Args:
            instances: Instances object containing predictions
            img_id: Image ID

        Returns:
            list[dict]: List of detection/segmentation results in COCO format
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()  # type: ignore
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_mask = instances.has("pred_masks")
        if has_mask:
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]  # type: ignore
                for mask in instances.pred_masks
            ]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            if has_mask:
                result["segmentation"] = rles[k]
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However our predictions are floating point coordinates.
                # Therefore we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
        return results


class InstanceSegmentationEvaluator(DetectionEvaluator):
    """Evaluator for instance segmentation predictions."""

    def __init__(self, dataset_dict: DictDataset, distributed=True):
        super().__init__(
            dataset_dict,
            task="segm",
            distributed=distributed,
        )


def create_coco(images, categories, coco_dict, iou_type):
    """
    Create COCO API object from detection/segmentation data.

    Args:
        images: List of dicts with image metadata
        categories: List of dicts with category information
        coco_dict: List of annotations/predictions in COCO format
        iou_type: Type of evaluation - "bbox" or "segm"

    Returns:
        COCO: COCO API object containing the data
    """
    res = COCO()
    res.dataset["images"] = images.values()
    res.dataset["categories"] = categories

    anns = coco_dict

    if iou_type == "bbox":
        res.dataset["categories"] = copy.deepcopy(res.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            ann["area"] = bb[2] * bb[3] if "area" not in ann else ann["area"]
            ann["id"] = id + 1
            ann["iscrowd"] = 0 if "iscrowd" not in ann else ann["iscrowd"]
    elif iou_type == "segm":
        res.dataset["categories"] = copy.deepcopy(res.dataset["categories"])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            if isinstance(ann["segmentation"], list):
                rles = mask_util.frPyObjects(
                    ann["segmentation"],
                    images[ann["image_id"]]["height"],
                    images[ann["image_id"]]["width"],
                )
                rle = mask_util.merge(rles)
            elif isinstance(ann["segmentation"]["counts"], list):
                rle = mask_util.frPyObjects(
                    ann["segmentation"],
                    images[ann["image_id"]]["height"],
                    images[ann["image_id"]]["width"],
                )
            else:
                rle = ann["segmentation"]
            ann["area"] = mask_util.area(rle) if "area" not in ann else ann["area"]
            if "bbox" not in ann:
                ann["bbox"] = mask_util.toBbox(rle)
            ann["id"] = id + 1
            ann["iscrowd"] = 0 if "iscrowd" not in ann else ann["iscrowd"]

    res.dataset["annotations"] = anns
    res.createIndex()
    return res
