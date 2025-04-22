# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from focoos.data import utils
from focoos.data.mappers.mapper import DatasetEntry
from focoos.data.transforms import augmentation as A
from focoos.data.transforms import transform as T
from focoos.structures import BoxMode, Instances

from .mapper import DatasetMapper


@dataclass
class DetectionDatasetDict(DatasetEntry):
    file_name: Optional[str] = None
    image_id: Optional[int] = None


class DetectionDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: Sequence[Union[A.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        recompute_boxes: bool = False,
    ):
        """
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = A.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        use_bbox = len(dataset_dict["annotations"]) > 0 and "bbox" in dataset_dict["annotations"][0]
        use_mask = len(dataset_dict["annotations"]) > 0 and "segmentation" in dataset_dict["annotations"][0]
        use_bbox = True if not (use_mask or use_bbox) else use_bbox

        # USER: Implement additional transformations if you have other types of data
        if self.is_train:
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            # we don't augment the boxes if we are in inference mode
            annos = [obj for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]

        instances: Instances = utils.annotations_to_instances(annos, image_shape)
        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes and self.use_instance_mask:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        instances = utils.filter_empty_instances(instances, by_box=use_bbox, by_mask=use_mask)

        if self.use_instance_mask:
            h, w = instances.image_size
            if hasattr(instances, "gt_masks"):  # Handle Images without annotations
                instances.gt_masks = instances.gt_masks.tensor
            else:
                instances.gt_masks = torch.zeros(0, h, w)

        dataset_dict["instances"] = instances

    def __call__(self, dataset_dict: dict) -> DetectionDatasetDict:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            DetectionDatasetDict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        self.check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict and self.is_train:
            # filter crowd annotations
            annotations = [obj for obj in dataset_dict["annotations"] if obj.get("iscrowd", 0) == 0]
            if len(annotations) > 0 and "bbox" in annotations[0]:
                boxes = []
                for annotation in annotations:
                    boxes.append(
                        BoxMode.convert(
                            annotation["bbox"],
                            annotation["bbox_mode"],
                            BoxMode.XYXY_ABS,
                        )
                    )
                # clip transformed bbox to image size
                boxes = np.array([boxes])[0].clip(min=0)
            else:
                boxes = None
        else:
            annotations = None
            boxes = None

        # we don't augment the boxes if we are in inference mode
        aug_input = A.AugInput(image, boxes=boxes)
        transforms = self.augmentations(aug_input)
        # we don't collect boxes but we recompute the transforms at the end
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict and self.is_train:
            # there is a problem here with image_shape (annotations are not transformed)
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return DetectionDatasetDict(
            image=dataset_dict["image"],
            height=dataset_dict["height"],
            width=dataset_dict["width"],
            file_name=dataset_dict["file_name"],
            image_id=dataset_dict["image_id"],
            instances=dataset_dict.get("instances", None),
        )


class InstanceDatasetMapper(DetectionDatasetMapper):
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[A.Augmentation, T.Transform]],
        image_format: str,
        recompute_boxes: bool = False,
    ):
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            use_instance_mask=True,
            recompute_boxes=recompute_boxes,
        )
