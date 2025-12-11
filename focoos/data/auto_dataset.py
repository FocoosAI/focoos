import os
from typing import List, Optional, Tuple, Union

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.datasets.map_dataset import MapDataset
from focoos.data.default_aug import DatasetAugmentations
from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetMapper
from focoos.data.mappers.detection_dataset_mapper import DetectionDatasetMapper
from focoos.data.mappers.keypoint import KeypointDatasetMapper
from focoos.data.mappers.mapper import DatasetMapper
from focoos.data.mappers.semantic_dataset_mapper import SemanticDatasetMapper
from focoos.data.transforms import augmentation as A
from focoos.data.transforms import transform as T
from focoos.ports import (
    DATASETS_DIR,
    DatasetLayout,
    DatasetSplitType,
    Task,
)
from focoos.utils.logger import get_logger
from focoos.utils.system import (
    check_folder_exists,
    extract_archive,
    is_inside_sagemaker,
)

logger = get_logger("AutoDataset")


class AutoDataset:
    def __init__(
        self,
        dataset_name: str,
        task: Task,
        layout: DatasetLayout,
        datasets_dir: str = DATASETS_DIR,
    ):
        self.task = task
        self.layout = layout
        self.datasets_dir = datasets_dir
        self.dataset_name = dataset_name

        if self.layout is not DatasetLayout.CATALOG:
            dataset_path = os.path.join(self.datasets_dir, dataset_name)
        else:
            dataset_path = self.datasets_dir

        if dataset_path.endswith(".zip") or dataset_path.endswith(".gz"):
            # compressed path: datasets_root_dir/dataset_compressed/{dataset_name}.zip
            # _dest_path = os.path.join(self.datasets_root_dir, dataset_name.split(".")[0])
            assert not (self.layout == DatasetLayout.CATALOG and not is_inside_sagemaker()), (
                "Catalog layout does not support compressed datasets externally to Sagemaker."
            )
            if self.layout == DatasetLayout.CATALOG:
                dataset_path = extract_archive(dataset_path)
                logger.info(f"Extracted archive: {dataset_path}, {os.listdir(dataset_path)}")
            else:
                dataset_name = dataset_name.split(".")[0]
                _dest_path = os.path.join(self.datasets_dir, dataset_name)
                dataset_path = extract_archive(dataset_path, _dest_path)
                logger.info(f"Extracted archive: {dataset_path}, {os.listdir(dataset_path)}")

        self.dataset_path = str(dataset_path)
        self.dataset_name = dataset_name
        logger.info(
            f"🔄 Loading dataset {self.dataset_name}, 📁 Dataset Path: {self.dataset_path}, 🗂️ Dataset Layout: {self.layout}"
        )

    def _load_split(self, dataset_name: str, split: DatasetSplitType) -> DictDataset:
        if self.layout == DatasetLayout.CATALOG:
            return DictDataset.from_catalog(ds_name=dataset_name, split_type=split, root=self.dataset_path)
        else:
            ds_root = self.dataset_path
            if not check_folder_exists(ds_root):
                raise FileNotFoundError(f"Dataset {ds_root} not found")
            split_path = self._get_split_path(dataset_root=ds_root, split_type=split)
            if self.layout == DatasetLayout.ROBOFLOW_SEG:
                return DictDataset.from_roboflow_seg(ds_dir=split_path, task=self.task, split_type=split)
            elif self.layout == DatasetLayout.CLS_FOLDER:
                return DictDataset.from_folder(root_dir=split_path, split_type=split)
            elif self.layout == DatasetLayout.ROBOFLOW_COCO:
                return DictDataset.from_roboflow_coco(ds_dir=split_path, task=self.task, split_type=split)
            else:  # Focoos
                raise NotImplementedError(f"Dataset layout {self.layout} not implemented")

    def _load_mapper(
        self,
        augs: List[Union[A.Augmentation, T.Transform]],
        is_validation_split: bool,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> DatasetMapper:
        if self.task == Task.SEMSEG:
            return SemanticDatasetMapper(
                image_format="RGB",
                ignore_label=255,
                augmentations=augs,
                is_train=not is_validation_split,
                resolution=resolution,
            )
        elif self.task == Task.DETECTION:
            return DetectionDatasetMapper(
                image_format="RGB",
                is_train=not is_validation_split,
                augmentations=augs,
                resolution=resolution,
            )
        elif self.task == Task.INSTANCE_SEGMENTATION:
            return DetectionDatasetMapper(
                image_format="RGB",
                is_train=not is_validation_split,
                augmentations=augs,
                use_instance_mask=True,
                resolution=resolution,
            )
        elif self.task == Task.CLASSIFICATION:
            return ClassificationDatasetMapper(
                image_format="RGB",
                is_train=not is_validation_split,
                augmentations=augs,
                resolution=resolution,
            )
        elif self.task == Task.KEYPOINT:
            return KeypointDatasetMapper(
                image_format="RGB",
                augmentations=augs,
                is_train=not is_validation_split,
                resolution=resolution,
                # keypoint_hflip_indices=np.array(keypoint_hflip_indices),
            )
        else:
            raise NotImplementedError(f"Task {self.task} not found in autodataset _load_mapper()")

    def _get_split_path(self, dataset_root: str, split_type: DatasetSplitType) -> str:
        if split_type == DatasetSplitType.TRAIN:
            possible_names = ["train", "training"]
            for name in possible_names:
                split_path = os.path.join(dataset_root, name)
                if check_folder_exists(split_path):
                    return split_path
            raise FileNotFoundError(f"Train split not found in {dataset_root}")
        elif split_type == DatasetSplitType.VAL:
            possible_names = ["valid", "val", "validation"]
            for name in possible_names:
                split_path = os.path.join(dataset_root, name)
                if check_folder_exists(split_path):
                    return split_path
            raise FileNotFoundError(f"Validation split not found in {dataset_root}")
        else:
            raise ValueError(f"Invalid split type: {split_type}")

    def get_split(
        self,
        augs: DatasetAugmentations,
        split: DatasetSplitType = DatasetSplitType.TRAIN,
    ) -> MapDataset:
        """
        Generate a dataset for a given dataset name with optional augmentations.

        Parameters:
            augs (DatasetAugmentations): Augmentations configuration.
                Resolution will be automatically extracted from this object.
            split (DatasetSplitType): Dataset split type (TRAIN or VAL).

        Returns:
            MapDataset: A DictDataset with DatasetMapper for training.
        """
        dict_split = self._load_split(dataset_name=self.dataset_name, split=split)
        assert dict_split.metadata.num_classes > 0, "Number of dataset classes must be greater than 0"

        # Extract resolution and augmentations from DatasetAugmentations
        resolution = augs.resolution
        augs_list = augs.get_augmentations()

        return MapDataset(
            dataset=dict_split,
            mapper=self._load_mapper(
                augs=augs_list,
                is_validation_split=(split == DatasetSplitType.VAL),
                resolution=resolution,
            ),
        )  # type: ignore
