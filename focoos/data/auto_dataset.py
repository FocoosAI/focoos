import os
from typing import List

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.datasets.map_dataset import MapDataset
from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetMapper
from focoos.data.mappers.detection_dataset_mapper import DetectionDatasetMapper
from focoos.data.mappers.mapper import DatasetMapper
from focoos.data.mappers.semantic_dataset_mapper import SemanticDatasetMapper
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

logger = get_logger(__name__)


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

        # if is_inside_sagemaker():
        #     # non compressed path: /opt/ml/input/data/dataset (there's not dataset name)
        #     # compressed path: /opt/ml/input/data/dataset_compressed/{dataset_name}.zip
        #     logger.info("Running inside Sagemaker: True")
        #     dataset_path = get_sgm_dataset_path(layout, self.datasets_root_dir)
        # else:
        #     # non compressed path: datasets_root_dir/dataset_name if not catalog
        #     #                      datasets_root_dir              if catalog
        #     # compressed path: datasets_root_dir/dataset_name.zip if not catalog
        #     #                  NOT SUPPORTED if catalog
        #     if self.layout is not DatasetLayout.CATALOG:
        #         dataset_path = os.path.join(self.datasets_root_dir, dataset_name)
        #     else:
        #         dataset_path = self.datasets_root_dir
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
            f"✅ Dataset name: {self.dataset_name}, Dataset Path: {self.dataset_path}, Dataset Layout: {self.layout}"
        )

    def _load_split(self, dataset_name: str, split: DatasetSplitType) -> DictDataset:
        if self.layout == DatasetLayout.CATALOG:
            return DictDataset.from_catalog(ds_name=dataset_name, split=split, root=self.dataset_path)
        else:
            ds_root = self.dataset_path
            if not check_folder_exists(ds_root):
                raise FileNotFoundError(f"Dataset {ds_root} not found")
            split_path = self._get_split_path(dataset_root=ds_root, split_type=split)
            if self.layout == DatasetLayout.ROBOFLOW_SEG:
                return DictDataset.from_roboflow_seg(ds_dir=split_path, task=self.task)
            elif self.layout == DatasetLayout.CLS_FOLDER:
                return DictDataset.from_folder(root_dir=split_path)
            # elif self.layout == DatasetLayout.SUPERVISELY:
            #     return DictDataset.from_supervisely(ds_dir=split_path, task=self.task)
            elif self.layout == DatasetLayout.ROBOFLOW_COCO:
                return DictDataset.from_roboflow_coco(ds_dir=split_path, task=self.task)
            else:  # Focoos
                raise NotImplementedError(f"Dataset layout {self.layout} not implemented")

    def _load_mapper(
        self,
        augs: List[T.Transform],
        is_validation_split: bool,
    ) -> DatasetMapper:
        if self.task == Task.SEMSEG:
            return SemanticDatasetMapper(
                image_format="RGB",
                ignore_label=255,
                augmentations=augs,
                is_train=not is_validation_split,
            )
        elif self.task == Task.DETECTION:
            return DetectionDatasetMapper(
                image_format="RGB",
                is_train=not is_validation_split,
                augmentations=augs,
            )
        elif self.task == Task.INSTANCE_SEGMENTATION:
            return DetectionDatasetMapper(
                image_format="RGB",
                is_train=not is_validation_split,
                augmentations=augs,
                use_instance_mask=True,
            )
        elif self.task == Task.CLASSIFICATION:
            return ClassificationDatasetMapper(
                image_format="RGB",
                is_train=not is_validation_split,
                augmentations=augs,
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
        augs: List[T.Transform],
        split: DatasetSplitType = DatasetSplitType.TRAIN,
    ) -> MapDataset:
        """
        Generate a dataset for a given dataset name with optional augmentations.

        Parameters:
            short_edge_length (int): The length of the shorter edge of the images.
            max_size (int): The maximum size of the images.
            extra_augs (List[Transform]): Extra augmentations to apply.

        Returns:
            MapDataset: A DictDataset with DatasetMapper for training.
        """

        return MapDataset(
            dataset=self._load_split(dataset_name=self.dataset_name, split=split),
            mapper=self._load_mapper(
                augs=augs,
                is_validation_split=(split == DatasetSplitType.VAL),
            ),
        )  # type: ignore
