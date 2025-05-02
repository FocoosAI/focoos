# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import numpy as np
import torch

from focoos.data import utils
from focoos.data.transforms import augmentation as A
from focoos.ports import DatasetEntry
from focoos.structures import BitMasks, Boxes, Instances

from .mapper import DatasetMapper


class PanopticDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        bounding_box=False,
    ):
        """
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
            bounding_box: compute and return bounding boxes for the masks
        """
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
        )
        self.bounding_box = bounding_box
        self.ignore_label = ignore_label

    def __call__(self, dataset_dict: dict) -> DatasetEntry:
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        self.check_image_size(dataset_dict, image)

        image, transforms = A.apply_augmentations(self.augmentations, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            if self.is_train:
                raise ValueError(
                    "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                        dataset_dict["file_name"]
                    )
                )
            pan_seg_gt = None
            segments_info = None

        # apply the same transformation to panoptic segmentation
        if pan_seg_gt is not None:
            try:
                from panopticapi.utils import rgb2id
            except ImportError:
                raise ImportError("panopticapi is not installed. Please install it with `pip install panopticapi`.")

            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            pan_seg_gt = rgb2id(pan_seg_gt)
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Prepare per-category binary masks
        if pan_seg_gt is not None:
            pan_seg_gt = pan_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor
                instances.gt_boxes = (
                    masks.get_bounding_boxes() if self.bounding_box else Boxes(torch.zeros((len(masks), 4)))
                )

            dataset_dict["instances"] = instances

        return DatasetEntry(
            image=dataset_dict["image"],
            height=dataset_dict["height"],
            width=dataset_dict["width"],
            file_name=dataset_dict["file_name"],
            image_id=dataset_dict["image_id"],
            instances=dataset_dict.get("instances", None),
        )
