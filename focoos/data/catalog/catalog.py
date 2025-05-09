import os
from dataclasses import dataclass
from typing import Optional

from focoos.data.catalog.utils import load_coco_json, load_sem_seg
from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.utils import filter_images_with_only_crowd_annotations
from focoos.ports import (
    DATASETS_DIR,
    DatasetMetadata,
    DatasetSplitType,
    Task,
)

# JSON FILE FORMAT:
# {
#    "categories": [
#     {
#         "id": 0,
#         "name": "Name",
#         "supercategory": "SuperName"
#     },...
#     ]
#     "images": [
#     {
#         "id": 0,
#         "file_name": "image_name",
#         "height": 640,
#         "width": 640,
#     }, ... ]
#     "annotations": [ # for detection
#     {
#         "id": 0,
#         "image_id": 0,
#         "category_id": 4,
#         "bbox": [
#             233,
#             307,
#             71,
#             69
#         ],
#         "area": 4899,
#         "segmentation": list of points or bitmask, # for instance seg
#         "iscrowd": 0
#     },
#     "annotations": [ # for semantic
#     {
#         "image_id": 0,
#         "file_name": "annotation.png"
#         "segments_info": [ { as bbox }, ... ]  # for panoptic
#     },


@dataclass
class CatalogSplit:
    image_root: str
    json_file: str
    gt_root: Optional[str] = None  # only for semantic/panoptic
    filter_empty: bool = True


@dataclass
class CatalogDataset:
    name: str
    task: Task
    train_split: CatalogSplit
    val_split: CatalogSplit
    test_split: Optional[CatalogSplit] = None


CATALOG = [
    CatalogDataset(
        name="ade20k_semseg",
        task=Task.SEMSEG,
        train_split=CatalogSplit(
            image_root="ADEChallengeData2016/images/training",
            gt_root="ADEChallengeData2016/annotations_detectron2/training",
            json_file="ADEChallengeData2016/ade20k_semseg_train.json",  # trick to get classes
        ),
        val_split=CatalogSplit(
            image_root="ADEChallengeData2016/images/validation",
            gt_root="ADEChallengeData2016/annotations_detectron2/validation",
            json_file="ADEChallengeData2016/ade20k_semseg_val.json",  # trick to get classes
        ),
    ),
    CatalogDataset(
        name="voc_semseg",
        task=Task.SEMSEG,
        train_split=CatalogSplit(
            image_root="PascalVOC12",
            gt_root="PascalVOC12",
            json_file="PascalVOC12/train.json",
        ),
        val_split=CatalogSplit(
            image_root="PascalVOC12",
            gt_root="PascalVOC12",
            json_file="PascalVOC12/val.json",
        ),
    ),
    CatalogDataset(
        name="ade20k_instance",
        task=Task.INSTANCE_SEGMENTATION,
        train_split=CatalogSplit(
            image_root="ADEChallengeData2016/images/training",
            json_file="ADEChallengeData2016/ade20k_instance_train.json",
        ),
        val_split=CatalogSplit(
            image_root="ADEChallengeData2016/images/validation",
            json_file="ADEChallengeData2016/ade20k_instance_val.json",
            filter_empty=False,
        ),
    ),
    # CatalogDataset(
    #     name="ade20k_panoptic",
    #     task=Task.,
    #     train_split=CatalogSplit(
    #         image_root="ADEChallengeData2016/images/training",
    #         gt_root="ADEChallengeData2016/ade20k_panoptic_train",
    #         json_file="ADEChallengeData2016/ade20k_panoptic_train.json",
    #     ),
    #     val_split=CatalogSplit(
    #         image_root="ADEChallengeData2016/images/validation",
    #         gt_root="ADEChallengeData2016/ade20k_panoptic_val",
    #         json_file="ADEChallengeData2016/ade20k_panoptic_val.json",
    #     ),
    # ),
    CatalogDataset(
        name="coco_2017_det",
        task=Task.DETECTION,
        train_split=CatalogSplit(
            image_root="coco/train2017",
            json_file="coco/annotations/instances_train2017.json",
        ),
        val_split=CatalogSplit(
            image_root="coco/val2017",
            json_file="coco/annotations/instances_val2017.json",
            filter_empty=False,
        ),
    ),
    CatalogDataset(
        name="coco_2017_instance",
        task=Task.INSTANCE_SEGMENTATION,
        train_split=CatalogSplit(
            image_root="coco/train2017",
            json_file="coco/annotations/instances_train2017.json",
        ),
        val_split=CatalogSplit(
            image_root="coco/val2017",
            json_file="coco/annotations/instances_val2017.json",
            filter_empty=False,
        ),
    ),
    # CatalogDataset(
    #     name="coco_2017_panoptic",
    #     task=Task.PANSEG,
    #     train_split=CatalogSplit(
    #         image_root="coco/train2017",
    #         gt_root="coco/annotations/panoptic_train2017",
    #         json_file="coco/annotations/panoptic_train2017.json",
    #     ),
    #     val_split=CatalogSplit(
    #         image_root="coco/val2017",
    #         gt_root="coco/annotations/panoptic_val2017",
    #         json_file="coco/annotations/panoptic_val2017.json",
    #     ),
    # ),
    CatalogDataset(
        name="object365",
        task=Task.DETECTION,
        train_split=CatalogSplit(
            image_root="object365/train",
            json_file="object365/train/_annotations.coco.json",
        ),
        val_split=CatalogSplit(
            image_root="object365/val",
            json_file="object365/val/_annotations.coco.json",
            filter_empty=False,
        ),
    ),
]


def _load_dataset_split(
    split_name: str,
    split: CatalogSplit,
    task: Task,
    root=DATASETS_DIR,
) -> DictDataset:
    """
    This function can be used for loading datasets outside the catalog but with the same format
    """

    def get_path(root, path):
        return os.path.join(root, path)

    json_file_path = get_path(root, split.json_file)
    image_root_path = get_path(root, split.image_root)
    gt_root_path = get_path(root, split.gt_root) if split.gt_root else None

    metadata = DatasetMetadata(
        name=split_name,
        num_classes=0,  # will be overridden
        json_file=json_file_path,
        image_root=image_root_path,
        task=task,
    )

    if task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
        dataset_dict = load_coco_json(json_file_path, image_root_path, metadata, task=task)
        if split.filter_empty:
            dataset_dict = filter_images_with_only_crowd_annotations(dataset_dicts=dataset_dict)
    elif task == Task.SEMSEG:
        if not gt_root_path:
            raise ValueError(f"Internal Error: gt_root missing from dataset {split_name}.")
        metadata.sem_seg_root = gt_root_path
        metadata.ignore_label = 255
        dataset_dict = load_sem_seg(
            gt_root=gt_root_path,
            image_root=image_root_path,
            json_file=json_file_path,
            metadata=metadata,
        )
        # elif task == Task.PANSEG:
        #     if not gt_root_path:
        #         raise ValueError(f"Internal Error: gt_root missing from dataset {split_name}.")
        #     metadata.panoptic_root = gt_root_path
        # dataset_dict = load_coco_panoptic_json(json_file_path, image_root_path, gt_root_path, metadata)
    else:
        raise ValueError(f"Unknown task {task}")

    metadata.count = len(dataset_dict)
    return DictDataset(dataset_dict, task=task, metadata=metadata)


def get_dataset_split(name: str, split: DatasetSplitType, datasets_root=DATASETS_DIR) -> DictDataset:
    """
    Load a dataset split from the catalog.
    """
    dataset_names = [ds.name for ds in CATALOG]
    if name not in dataset_names:
        raise ValueError(f"Dataset {name} not found. Available datasets: {dataset_names}")

    ds = next(ds for ds in CATALOG if ds.name == name)
    if split == DatasetSplitType.TRAIN:
        entry = ds.train_split
        split_name = name
    elif split == DatasetSplitType.VAL:
        entry = ds.val_split
        split_name = name
    else:
        raise ValueError(f"Unknown split {split}")

    return _load_dataset_split(split_name, entry, ds.task, datasets_root)
