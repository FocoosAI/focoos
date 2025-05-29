import concurrent.futures
import csv
import json
import os
import random
from copy import copy
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset

from focoos.data.datasets.serialize import TorchSerializedDataset
from focoos.data.transforms.resize_short_length import resize_shortest_length
from focoos.ports import (
    DatasetMetadata,
    DatasetSplitType,
    DetectronDict,
    Task,
)
from focoos.utils.cmap_builder import cmap_builder
from focoos.utils.logger import get_logger
from focoos.utils.system import list_files_with_extensions


def remove_none_from_dict(data):
    return {k: v for (k, v) in data if v is not None}


class DictDataset(Dataset):
    def __init__(
        self,
        dicts: list[DetectronDict],
        task: Task,
        metadata: DatasetMetadata,
        serialize: bool = True,
    ):
        self.task: Task = task
        self.metadata: DatasetMetadata = metadata
        # self.dicts: list[DetectronDict] = dicts
        # assemble detectron standard dict
        self.logger = get_logger(__name__)
        self.logger.info(
            f"[Focoos-DictDataset] dataset {self.metadata.name} loaded. len: {self.metadata.count}, classes:{self.metadata.num_classes} ,{self.metadata.image_root}"
        )
        for i, d in enumerate(dicts):
            d.image_id = i

        self.dicts: Union[TorchSerializedDataset, list[DetectronDict]] = (
            TorchSerializedDataset(dicts) if serialize else dicts
        )

    def __getitem__(self, index) -> dict:
        entry = self.dicts[index]
        return asdict(entry, dict_factory=remove_none_from_dict)

    def __len__(self):
        return len(self.dicts)

    def store_coco_roboflow_format(self, output_dir: str):
        """
        Store the dataset in COCO format.
        """

        def compute_area_seg(seg):
            # let's assume the format is Polygon
            # Convert list of points to numpy array
            points = np.array(seg[0]).reshape(-1, 2)

            # Calculate area using shoelace formula
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return area

        def compute_area_box(bbox):
            return bbox[2] * bbox[3]  # we assume format is XYWH

        json_dict = {
            "info": {
                "year": "2025",
                "version": "1",
                "description": "Exported from FocoosAi",
            },
            "categories": [
                {
                    "id": 0,
                    "name": "custom_class",
                    "supercategory": "none",
                },
            ],
            "images": [],
            "annotations": [],
        }
        if self.metadata.thing_classes is None:
            raise ValueError("thing_classes is None")
        for i, cls in enumerate(self.metadata.thing_classes):
            json_dict["categories"].append(
                {
                    "id": i + 1,
                    "name": cls,
                    "supercategory": "custom_class",
                },
            )
        annotation_idx = 1
        for i, data in enumerate(self.dicts):  # type: ignore
            json_dict["images"].append(
                {
                    "id": data.image_id,
                    "file_name": data.file_name.split("/")[-1],
                    "height": data.height,
                    "width": data.width,
                }
            )
            for ann in data.annotations:
                use_seg = "segmentation" in ann
                area = compute_area_seg(ann["segmentation"]) if use_seg else compute_area_box(ann["bbox"])
                obj = {
                    "id": annotation_idx,
                    "image_id": data.image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": area,  # to compute
                    "iscrowd": ann["iscrowd"],
                }
                if use_seg:
                    obj["segmentation"] = ann["segmentation"]

                json_dict["annotations"].append(obj)
                annotation_idx += 1

        with open(os.path.join(output_dir, "_annotations.coco.json"), "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def from_catalog(cls, ds_name: str, split: DatasetSplitType, root: str):
        from focoos.data.catalog.catalog import get_dataset_split

        # importing catalog here is the only way to avoid circular input
        return get_dataset_split(name=ds_name, split=split, datasets_root=root)

    @classmethod
    def from_folder(cls, root_dir: str, split: Optional[DatasetSplitType] = None):
        """
        Create a dataset from a folder structure where categories are subfolders
        and images belonging to each category are inside these subfolders.

        Args:
            root_dir (str): Path to the root directory containing category subfolders
            split (Optional[DatasetSplitType]): Dataset split type (train, val, test)
                If provided, looks for the split in the root_dir/split directory

        Returns:
            ClassificationDataset: A dataset containing the images and their class labels
        """
        logger = get_logger(__name__)

        # If split is provided, update the root directory to include the split
        if split is not None:
            root_dir = os.path.join(root_dir, split.value)
            if not os.path.exists(root_dir):
                raise ValueError(f"Split directory {root_dir} does not exist")

        # Get all category directories
        category_dirs = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ]

        if not category_dirs:
            raise ValueError(f"No category directories found in {root_dir}")

        # Sort categories for deterministic ordering
        category_dirs.sort()

        # Create a mapping from category name to class ID
        class_to_idx = {cls_name: i for i, cls_name in enumerate(category_dirs)}

        # Initialize lists to store dataset entries
        dataset_dicts = []

        # Process each category
        for category in category_dirs:
            category_path = os.path.join(root_dir, category)
            class_id = class_to_idx[category]

            # Find all image files in the category folder
            image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
            image_files = list_files_with_extensions(category_path, image_extensions)

            # Create a dataset entry for each image
            for img_path in image_files:
                try:
                    # Open image to get dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size

                    # Create dataset entry
                    entry = DetectronDict(
                        file_name=str(img_path),
                        height=height,
                        width=width,
                        # Store the class label as an annotation for compatibility with the DictDataset structure
                        annotations=[{"category_id": class_id, "iscrowd": 0}],
                    )
                    dataset_dicts.append(entry)
                except (IOError, OSError) as e:
                    logger.warning(f"Error loading image {img_path}: {e}")

        # Create dataset metadata
        metadata = DatasetMetadata(
            num_classes=len(category_dirs),
            thing_classes=category_dirs,  # Use thing_classes for classification
            task=Task.CLASSIFICATION,
            count=len(dataset_dicts),
            name=Path(root_dir).name,
            image_root=root_dir,
        )

        logger.info(f"Created classification dataset with {len(dataset_dicts)} images and {len(category_dirs)} classes")

        return cls(dicts=dataset_dicts, task=Task.CLASSIFICATION, metadata=metadata)

    @classmethod
    def from_roboflow_coco(cls, ds_dir: str, task: Task):
        """
        ds_dir is up to the split.
        root/
            test/
                ..
            valid/
                ..
            train/
                _annotations.coco.json
                im0.jpeg
        """
        import pycocotools.mask as mask_util
        from pycocotools.coco import COCO

        from focoos.structures import BoxMode

        json_file = os.path.join(ds_dir, "_annotations.coco.json")
        coco_api = COCO(json_file)

        cat_ids = sorted(coco_api.getCatIds())
        for cat_id in cat_ids:  # remove class 0 if exists
            if cat_id <= 0:
                cat_ids.pop(cat_id)
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        id_map = {v: i for i, v in enumerate(cat_ids)}

        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

        imgs_anns = list(zip(imgs, anns))

        dataset_dicts = []

        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "area"]

        num_instances_without_valid_segmentation = 0

        for img_dict, anno_dict_list in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(ds_dir, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id

                assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                obj = {key: anno[key] for key in ann_keys if key in anno}
                if "bbox" in obj and len(obj["bbox"]) == 0:
                    raise ValueError(
                        f"One annotation of image {image_id} contains empty 'bbox' value! "
                        "This json does not have valid COCO format."
                    )
                is_crowd = obj.get("iscrowd", 0)
                if is_crowd == 1:
                    continue

                segm = anno.get("segmentation", None)
                if segm is not None and task == Task.INSTANCE_SEGMENTATION:  # either list[list[float]] or dict(RLE)
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                        else:
                            print("What happens here?")
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if id_map:
                    annotation_category_id = obj["category_id"]
                    try:
                        obj["category_id"] = id_map[annotation_category_id]
                    except KeyError as e:
                        raise KeyError(
                            f"Encountered category_id={annotation_category_id} "
                            "but this id does not exist in 'categories' of the json file."
                        ) from e

                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(DetectronDict(**record))

        metadata = DatasetMetadata(
            num_classes=len(thing_classes),
            thing_classes=thing_classes,
            task=task,
            count=len(dataset_dicts),
            name=Path(ds_dir).parent.stem,
            image_root=ds_dir,
            thing_dataset_id_to_contiguous_id=id_map,
            json_file=json_file,
        )

        return cls(dicts=dataset_dicts, task=task, metadata=metadata)

    @classmethod
    def from_segmentation(cls, ds_dir: str, task: Task, serialize: bool = True):
        """
        ds_dir is up to the split.
        root/
            test/
                ..
            valid/
                ..
            train/
                annotations.json
                your_format_here/

        JSON FORMAT:
        {
            "images": [
                {
                    "id": 0,
                    "file_name": "im0.jpeg",
                    "height": 1024,
                    "width": 1024
                },
            ],
            "annotations": [
                {
                   "image_id": 0,
                   "file_name": "im0.png",
                }
            ]
            "categories": [
                {
                    "id": 0,
                    "name": "custom_class",
                    "color": "none", [optional]
                    "is_thing": True,  [optional]
                },
            ]
        }
        """
        logger = get_logger(__name__)

        with open(os.path.join(ds_dir, "annotations.json")) as f:
            json_info = json.load(f)

        images = dict()
        for info in json_info["images"]:
            images[info["id"]] = info["file_name"]

        dataset_dicts = []
        for ann in json_info["annotations"]:
            image_id = ann["image_id"]

            image_file = os.path.join(ds_dir, images[image_id])
            label_file = os.path.join(ds_dir, ann["file_name"])

            dataset_dicts.append(DetectronDict(file_name=image_file, sem_seg_file_name=label_file, image_id=image_id))

        logger.info("Loaded {} images with semantic segmentation from {}".format(len(dataset_dicts), ds_dir))

        # This is only useful for metadata
        categories = json_info["categories"]
        # All the classes are stuff, only a subset is thing
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(categories):
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # Create dataset metadata
        metadata = DatasetMetadata(
            num_classes=len(categories),
            stuff_classes=[k["name"] for k in categories],
            _stuff_colors=[k["color"] for k in categories],
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            task=Task.SEMSEG,
            count=len(dataset_dicts),
            name=Path(ds_dir).name,
            image_root=ds_dir,
        )

        return cls(dicts=dataset_dicts, task=Task.SEMSEG, metadata=metadata, serialize=serialize)

    @classmethod
    def from_roboflow_seg(cls, ds_dir: str, task: Task):
        """
        root/
            test/
                ..
            valid/
                ..
            train/
                _classes.csv
                im0.jpeg
                im0_mask.png
        """
        print(f"ds_dir: {ds_dir}")
        im_files = []

        for im in list_files_with_extensions(base_dir=ds_dir, extensions=["jpg", "jpeg", "png"]):
            im = str(im)
            if not im.endswith("_mask.png"):
                im_files.append(im)

        classes = []

        cls_path = os.path.join(ds_dir, "_classes.csv")

        with open(cls_path, newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            next(reader, None)  # skip the headers
            for row in reader:
                classes.append(row[1].strip())
        dicts = []
        im_files.sort()

        for im in tqdm.tqdm(im_files):
            mask = im.replace(".jpg", "_mask.png")
            if not os.path.exists(mask):
                raise ValueError(f"Mask file {mask} does not exist")
            if Path(im).stem != Path(mask).stem.replace("_mask", ""):
                raise ValueError(f" {Path(im).stem} and {Path(mask).stem.replace('_mask', '')} mismatch")
            dicts.append(DetectronDict(file_name=im, sem_seg_file_name=mask))

        metadata = DatasetMetadata(
            stuff_classes=classes,
            num_classes=len(classes),
            task=task,
            name=Path(ds_dir).parent.stem,
            count=len(im_files),
            image_root=ds_dir,
        )

        return cls(dicts=dicts, task=task, metadata=metadata)

    def clone_resize_shortest_length(self, new_dir: str, new_shortest_length: int = 1024, max_size=2048):
        """
        Clone and resize DatasetDict images and masks to a new directory with a specified shortest length. and max size

        Parameters:
            new_dir (str): The directory path where the cloned and resized images and masks will be saved.
            new_shortest_length (int, optional): The new shortest length to resize the images and masks to. Defaults to 1024.
            max_size: The maximum size for the resized images and masks. Defaults to 2048.
        """
        logger = get_logger(__name__)
        logger.info("[START RESIZE] clone_resize_shortest_length ")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=150)
        os.makedirs(new_dir, exist_ok=True)
        # !TODO generalize for other task
        im_dir = os.path.join(new_dir, "img")
        mask_dir = os.path.join(new_dir, "mask")
        metadata_path = os.path.join(new_dir, "focoos_meta.json")
        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        orig_meta = self.metadata

        for data in tqdm.tqdm(self.dicts):  # type: ignore
            im_file = data.file_name
            mask_file = data.sem_seg_file_name
            pool.submit(
                resize_shortest_length,
                im_file,
                im_dir,
                new_shortest_length,
                max_size,
                False,
            )
            pool.submit(
                resize_shortest_length,
                mask_file,
                mask_dir,
                new_shortest_length,
                max_size,
                True,
            )
        pool.shutdown(wait=True)
        count = len(list_files_with_extensions(base_dir=im_dir, extensions=["png", "jpeg", "jpg"]))
        metadata = DatasetMetadata(
            count=count,
            num_classes=orig_meta.num_classes,
            task=orig_meta.task,
            thing_classes=orig_meta.thing_classes,
            stuff_classes=orig_meta.stuff_classes,
        )

        metadata.dump_json(metadata_path)
        logger.info("[END resize]")

    def get_annotated_sample(self, idx: int, resize: Optional[tuple] = None) -> Optional[Image.Image]:
        # !TODO generalize for other tasks
        if idx > len(self.dicts):
            return None
        else:
            cmap = cmap_builder()
            im_file = self.dicts[idx].file_name
            mask_file = self.dicts[idx].sem_seg_file_name
            if mask_file is None:
                self.logger.warning(f"Mask file {mask_file} is None for image {im_file}")
                return None
            mask_im = Image.open(mask_file)
            orig_im = Image.open(im_file).convert("RGB")
            mask = np.array(mask_im, dtype=np.uint8)
            output_colored = cmap[mask]

            out_img = Image.fromarray(output_colored)
            out_img = Image.blend(orig_im, out_img, 0.7)
            if resize:
                out_img = out_img.resize(size=resize)
            return out_img

    def split(self, ratio: float, shuffle: bool = True, seed: int = 42) -> Tuple["DictDataset", "DictDataset"]:
        random.seed(seed)
        _dicts = copy(self.dicts)

        if shuffle:
            random.shuffle(_dicts)  # type: ignore
        split_idx = int(len(_dicts) * ratio)
        split1 = _dicts[:split_idx]
        meta1 = DatasetMetadata(
            num_classes=self.metadata.num_classes,
            task=self.metadata.task,
            count=len(split1),
            thing_classes=self.metadata.thing_classes,
            stuff_classes=self.metadata.stuff_classes,
        )

        split2 = _dicts[split_idx:]
        meta2 = DatasetMetadata(
            num_classes=self.metadata.num_classes,
            task=self.metadata.task,
            count=len(split2),
            thing_classes=self.metadata.thing_classes,
            stuff_classes=self.metadata.stuff_classes,
        )

        return DictDataset(dicts=split1, task=self.metadata.task, metadata=meta1), DictDataset(
            dicts=split2, task=self.metadata.task, metadata=meta2
        )
