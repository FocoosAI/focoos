# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.mappers.semantic_dataset_mapper import SemanticSegmentationDatasetEntry
from focoos.trainer.evaluation.evaluator import DatasetEvaluator
from focoos.utils.distributed.comm import all_gather, is_main_process, synchronize
from focoos.utils.logger import get_logger

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


logger = get_logger(__name__)


def convert_boxes_to_gt_mask(boxes, classes, gt_image_size, output_image_size):
    gt_image_height, gt_image_width  = gt_image_size
    output_image_height, output_image_width = output_image_size
    mask = torch.zeros((output_image_height, output_image_width), dtype=torch.int32, device=boxes.device)
    
    for box, class_idx in zip(boxes, classes):
        x, y, w, h = box # XYWH_ABS format
        center_x, center_y = int(x + w/2), int(y + h/2)
        center_x = int(center_x * output_image_width / gt_image_width)
        center_y = int(center_y * output_image_height / gt_image_height)
        # classes are 0-indexed, so we need to add 1 to the class index, 0 will be background
        mask[center_y, center_x] = class_idx + 1 
    
    return mask


def load_image_into_numpy_array(
    filename: str,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with open(filename, "rb") as f:
        array = np.array(Image.open(f), dtype=dtype)
    return array


class CountingEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_dict: DictDataset,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
    ):
        """
        Args:s
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
        """
        self.dataset_dict = dataset_dict
        self.metadata = self.dataset_dict.metadata

        self._dataset_name = self.metadata.name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._eval_counter = 0 # DEBUG

        self._ignore_label = self.metadata.ignore_label
        if self.metadata.thing_classes is None:
            raise ValueError("thing_classes is None")
        self._num_classes = len(self.metadata.thing_classes)

        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = self.metadata.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = self.metadata.thing_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(self.metadata.thing_classes)

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs: list[SemanticSegmentationDatasetEntry], outputs: list[dict[str, torch.Tensor]]):
        """DEBUG"""
        if True:
            import os
            debug_dir = f"/home/ubuntu/focoos-1/notebooks/debug_outputs/eval_debug8test/eval_counter_{self._eval_counter}"
            os.makedirs(debug_dir, exist_ok=True)
        """DEBUG"""
        for input, output in zip(inputs, outputs):
            output = output["instances"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_annotations = self.dataset_dict[input.image_id]["annotations"]
            gt_boxes = torch.stack([torch.tensor(ann["bbox"]) for ann in gt_annotations]) # XYWH_ABS format
            gt_classes = torch.stack([torch.tensor(ann["category_id"]) for ann in gt_annotations])
            gt_image_size = self.dataset_dict[input.image_id]["height"], self.dataset_dict[input.image_id]["width"]
            gt_mask = convert_boxes_to_gt_mask(gt_boxes, gt_classes, 
                                               gt_image_size=gt_image_size, 
                                               output_image_size=output.shape
                                            )

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * torch.from_numpy(pred.reshape(-1)) + gt_mask.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input.file_name))
            
            """DEBUG"""
            if True:
                import os
                import shutil
                
                # Delete directory contents if it exists
                if os.path.exists(debug_dir):
                    if not hasattr(self, '_debug_dir_cleared'):
                        shutil.rmtree(debug_dir)
                        self._debug_dir_cleared = True
                        os.makedirs(debug_dir)
                
                    # Save the input image tensor
                    image_id = input.image_id
                    image_tensor = input.image
                    if image_tensor is not None:
                        image_save_path = os.path.join(debug_dir, f"{image_id}_image.pt")
                        torch.save(image_tensor, image_save_path)
                    # Save the ground truth mask
                    gt_mask_save_path = os.path.join(debug_dir, f"{image_id}_gt_mask.pt")
                    torch.save(gt_mask, gt_mask_save_path)
                    # Save the prediction mask
                    pred_save_path = os.path.join(debug_dir, f"{image_id}_pred_mask.pt")
                    torch.save(torch.from_numpy(pred), pred_save_path)
                    # print(f"DEBUG: tensors saved to : {debug_dir}")
            """DEBUG"""

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        results = OrderedDict({"sem_seg": res})
        logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert label in self._contiguous_id_to_dataset_id, "Label {} is not in the metadata info for {}".format(
                    label, self._dataset_name
                )
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {
                    "file_name": input_file_name,
                    "category_id": dataset_id,
                    "segmentation": mask_rle,
                }
            )
        return json_list
