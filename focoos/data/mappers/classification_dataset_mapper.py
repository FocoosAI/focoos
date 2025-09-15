import copy
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch

from focoos.data import utils
from focoos.data.mappers.mapper import DatasetMapper
from focoos.data.transforms import augmentation as A
from focoos.data.transforms import transform as T
from focoos.ports import DatasetEntry
from focoos.utils.logger import get_logger


@dataclass
class ClassificationDatasetDict(DatasetEntry):
    """
    Dataset dictionary for classification tasks.
    Extends the base DatasetEntry with fields needed for classification.
    """

    label: Optional[list[int]] = None


class ClassificationDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2/Focoos Dataset format,
    and maps it into a format used by classification models.

    It performs the following operations:
    1. Read the image from "file_name"
    2. Apply augmentations to the image
    3. Prepare image tensor and class label
    """

    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: Sequence[Union[A.Augmentation, T.Transform]],
        image_format: str = "RGB",
    ):
        """
        Args:
            is_train: Whether it's used in training or inference
            augmentations: A list of augmentations or transforms to apply
            image_format: An image format supported by PIL and OpenCV
        """
        super().__init__(
            is_train=is_train,
            augmentations=augmentations,  # type: ignore
            image_format=image_format,
        )
        self.logger = get_logger(__name__)
        mode = "training" if is_train else "inference"
        self.logger.info(f"[ClassificationDatasetMapper] Augmentations used in {mode}: {augmentations}")

    def __call__(self, dataset_dict: dict) -> ClassificationDatasetDict:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2/Focoos Dataset format.

        Returns:
            ClassificationDatasetDict: A format that contains the image and label
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # It will be modified by code below

        # Read image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        self.check_image_size(dataset_dict, image)

        # Extract class label from annotations
        label = []
        if "annotations" in dataset_dict and len(dataset_dict["annotations"]) > 0:
            # For classification, we take the first annotation's category_id as the label
            label = [ann.get("category_id", None) for ann in dataset_dict["annotations"]]

        # Apply augmentations
        aug_input = A.AugInput(image)
        self.augmentations(aug_input)  # apply augmentations in place, no need to return
        image = aug_input.image
        if image is None:
            raise ValueError(f"Image is None for {dataset_dict['file_name']}")

        # Convert image to tensor format (C, H, W)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Create the return object
        return ClassificationDatasetDict(
            image=dataset_dict["image"],
            height=dataset_dict["height"],
            width=dataset_dict["width"],
            file_name=dataset_dict["file_name"],
            image_id=dataset_dict.get("image_id", None),
            label=label,
        )
