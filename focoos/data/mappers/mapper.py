import logging
from typing import List, Union

from focoos.data.transforms import augmentation as A
from focoos.data.transforms import transform as T
from focoos.ports import DatasetEntry
from focoos.utils.logger import log_first_n


class DatasetMapper:
    def __init__(
        self,
        is_train=True,
        *,
        augmentations: List[Union[A.Augmentation, T.Transform]],
        image_format: str,
    ):
        self.is_train = is_train
        self.augmentations = A.AugmentationList(augmentations)
        self.image_format = image_format

    def check_image_size(self, dataset_dict, image):
        expected_wh = None
        if "width" in dataset_dict or "height" in dataset_dict:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
        if not expected_wh or image_wh != expected_wh:
            if expected_wh:
                log_first_n(
                    logging.WARNING,
                    "Image size is different from the one in the annotations.",
                    n=1,
                )
            dataset_dict["width"] = image.shape[1]
            dataset_dict["height"] = image.shape[0]

    def __call__(self, dataset_dic: dict) -> DatasetEntry:
        """
        Args:
            dataset_dict (DetectronDict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            DatasetEntry: an object containing the image, annotations and metadata
        """
        raise NotImplementedError("This is an abstract class, never use DatasetMapper directly")
