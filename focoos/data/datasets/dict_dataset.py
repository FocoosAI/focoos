import concurrent.futures
import csv
import json
import logging
import os
import random
import shutil
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
        self.logger = logging.getLogger(__name__)
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
        from focoos.data.catalog import get_dataset_split

        # importing catalog here is the only way to avoid circular input
        return get_dataset_split(name=ds_name, split=split, datasets_root=root)

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

    # !TODO: reimplement this without using supervisely
    # @classmethod
    # def from_supervisely(cls, ds_dir: str, task: Task):
    #     """
    #     A method to create an instance of the class from supervisely metadata.

    #     Args:
    #         ds_dir (str): The directory of the supervisely dataset.
    #         task (FocoosTasks): The task type.

    #     Returns:
    #         An instance of the class created from the supervisely metadata.
    #     """
    #     # check if exist supervisely metadata in parent directory (supervisely project)
    #     logger = logging.getLogger(__name__)
    #     im_path = os.path.join(ds_dir, "img")

    #     # supervisely json annotation
    #     ann_path = os.path.join(ds_dir, "ann")
    #     mask_path = os.path.join(ds_dir, "mask")
    #     im_files = []
    #     mask_files = []
    #     classes = []
    #     dicts = []
    #     # load project meta
    #     sly_meta = None
    #     parent_dir = os.path.dirname(ds_dir)
    #     meta_path = os.path.join(parent_dir, "meta.json")
    #     if not os.path.exists(meta_path):
    #         # check in ds directory
    #         meta_path = os.path.join(ds_dir, "meta.json")

    #         if not os.path.exists(meta_path):
    #             raise ValueError("Supervisely metadata not found")

    #     with open(meta_path) as _meta:
    #         sly_meta = ProjectMeta.from_json(json.load(_meta))
    #         classes = [obj_cls.name for obj_cls in sly_meta.obj_classes]
    #         logger.info(f"Loaded supervisely metadata, classes: {classes}")

    #         # ann = SlyAnnotation.from_json(meta_dict)
    #     for im in list_files_with_extensions(base_dir=im_path, extensions=["jpg", "jpeg", "png"]):
    #         im_files.append(im)
    #     logger.info(f"{len(im_files)} images")
    #     for mask in list_files_with_extensions(base_dir=mask_path, extensions=["png"]):
    #         mask_files.append(mask)

    #     logger.info(f"{len(mask_files)} masks")
    #     if len(mask_files) == 0:
    #         logger.info("No png mask found...generating from supervisely annotation..")
    #         # pool = Pool(processes=30)
    #         pool = concurrent.futures.ThreadPoolExecutor(max_workers=150)

    #         os.makedirs(mask_path, exist_ok=True)
    #         for ann in tqdm.tqdm(list_files_with_extensions(base_dir=ann_path, extensions=["json"])):
    #             name = f"{Path(ann).stem}.mask.png"
    #             _path = os.path.join(mask_path, name)
    #             ann = Annotation.from_json(json.load(open(ann)), project_meta=sly_meta)
    #             pool.submit(sly_ann_to_bitmap_mask, ann, _path, sly_meta, 256)
    #             mask_files.append(_path)
    #         pool.shutdown(wait=True)
    #         # pool.close()
    #     im_files.sort()
    #     mask_files.sort()
    #     for im, mask in tqdm.tqdm(zip(im_files, mask_files)):
    #         if not Path(mask).stem.replace(".mask", "").startswith(Path(im).stem):
    #             raise ValueError(f" {Path(im).stem} and {Path(mask).stem.replace('.mask', '')} mismatch")
    #         dicts.append(DetectronDict(file_name=im, sem_seg_file_name=mask))

    #     metadata = DatasetMetadata(
    #         stuff_classes=classes,
    #         num_classes=len(classes),
    #         task=task,
    #         name=Path(ds_dir).parent.stem,
    #         count=len(im_files),
    #         image_root=ds_dir,
    #     )

    #     return cls(dicts=dicts, task=task, metadata=metadata)

    def clone_resize_shortest_length(self, new_dir: str, new_shortest_length: int = 1024, max_size=2048):
        """
        Clone and resize DatasetDict images and masks to a new directory with a specified shortest length. and max size

        Parameters:
            new_dir (str): The directory path where the cloned and resized images and masks will be saved.
            new_shortest_length (int, optional): The new shortest length to resize the images and masks to. Defaults to 1024.
            max_size: The maximum size for the resized images and masks. Defaults to 2048.
        """
        logger = logging.getLogger(__name__)
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

    def split(
        self,
        ratio: float,
        new_root: str,
        split1_name: str = "training",
        split2_name: str = "validation",
        shuffle: bool = True,
    ) -> Tuple[str, str]:
        random.seed(42)
        _dicts = copy(self.dicts)
        split1_path = os.path.join(new_root, split1_name)
        split2_path = os.path.join(new_root, split2_name)
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
        os.makedirs(split1_path, exist_ok=True)
        os.makedirs(split2_path, exist_ok=True)
        meta1.dump_json(os.path.join(split1_path, "focoos_meta.json"))
        meta2.dump_json(os.path.join(split2_path, "focoos_meta.json"))

        # copy files
        for split_path, split in [(split1_path, split1), (split2_path, split2)]:
            # create dirs
            im_path = os.path.join(split_path, "img")
            mask_path = os.path.join(split_path, "mask")
            os.makedirs(im_path, exist_ok=True)
            os.makedirs(mask_path, exist_ok=True)
            for data in split:
                im = data.file_name
                mask = data.sem_seg_file_name
                shutil.copy(im, im_path)
                shutil.copy(mask, mask_path)  # type: ignore

        return split1_path, split2_path
