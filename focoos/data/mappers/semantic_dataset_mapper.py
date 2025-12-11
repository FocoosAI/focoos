# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from focoos.data import utils
from focoos.data.transforms import augmentation as A
from focoos.ports import DatasetEntry
from focoos.structures import BitMasks, Instances
from focoos.utils.logger import get_logger

from .mapper import DatasetMapper


@dataclass
class SemanticSegmentationDatasetEntry(DatasetEntry):
    """
    Dataset entry for semantic segmentation evaluation.
    """

    sem_seg_file_name: Optional[str] = None


class SemanticDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

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
        image_format="RGB",
        ignore_label=255,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        """
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            resolution: Image resolution used for augmentations.
        """
        self.is_train = is_train
        self.augmentations = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.resolution = resolution

        logger = get_logger("SemSegMapper")
        mode = "train" if is_train else "val"
        augs_str = "\n - ".join([str(aug) for aug in augmentations])
        logger.info(
            f"\n =========== 🎨 {mode} augmentations =========== \n - {augs_str} \n============================================"
        )

    def __call__(self, dataset_dict: dict) -> SemanticSegmentationDatasetEntry:
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        self.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict and self.is_train:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None and self.is_train:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = A.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = A.apply_augmentations(self.augmentations, aug_input)

        if not hasattr(aug_input, "image") or aug_input.image is None:  # type: ignore
            raise ValueError(f"Image is None for {dataset_dict['file_name']}")
        image = aug_input.image  # type: ignore
        sem_seg_gt = aug_input.sem_seg if sem_seg_gt is not None else None  # type: ignore

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()

            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            classes = torch.tensor(classes, dtype=torch.int64)

            masks_np = []
            for class_id in classes:
                masks_np.append(sem_seg_gt == class_id)

            if len(masks_np) == 0:
                # Some image does not have annotation (all ignored)
                masks = BitMasks(torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])))
            else:
                masks = BitMasks(torch.stack([x.contiguous() for x in masks_np]))

            dataset_dict["instances"] = Instances(image_shape, classes=classes, masks=masks)

        return SemanticSegmentationDatasetEntry(
            image=dataset_dict["image"],
            height=dataset_dict["height"],
            width=dataset_dict["width"],
            file_name=dataset_dict["file_name"],
            image_id=dataset_dict["image_id"],
            instances=dataset_dict.get("instances", None),
            sem_seg_file_name=dataset_dict.get("sem_seg_file_name", None),
        )
