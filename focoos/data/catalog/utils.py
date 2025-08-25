import contextlib
import io
import json
import os

import pycocotools.mask as mask_util

from focoos.ports import DatasetMetadata, DetectronDict, Task
from focoos.structures import BoxMode
from focoos.utils.logger import get_logger
from focoos.utils.timer import Timer

logger = get_logger(__name__)


def load_sem_seg(
    gt_root,
    image_root,
    json_file,
    metadata: DatasetMetadata,
):
    with open(json_file) as f:
        json_info = json.load(f)

    images = dict()
    for info in json_info["images"]:
        images[info["id"]] = info["file_name"]

    dataset_dicts = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]

        image_file = os.path.join(image_root, images[image_id])
        label_file = os.path.join(gt_root, ann["file_name"])

        dataset_dicts.append(DetectronDict(file_name=image_file, sem_seg_file_name=label_file, image_id=image_id))

    logger.info("Loaded {} images with semantic segmentation from {}".format(len(dataset_dicts), image_root))

    # This is only useful for metadata
    categories = json_info["categories"]
    # All the classes are stuff, only a subset is thing
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(categories):
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    metadata.stuff_classes = [k["name"] for k in categories]
    metadata.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    metadata.num_classes = len(categories)
    if "color" in categories[0]:
        metadata.stuff_colors = [k["color"] for k in categories]

    return dataset_dicts


def load_coco_json(
    json_file,
    image_root,
    metadata: DatasetMetadata,
    task: str,
    extra_annotation_keys=None,
):
    from pycocotools.coco import COCO

    timer = Timer()
    # json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    keypoints = None
    keypoints_skeleton = None
    if len(cats) > 0 and cats[0].get("keypoints", None) is not None:
        keypoints = cats[0].get("keypoints", None)
        if cats[0].get("skeleton", None) is not None:
            keypoints_skeleton = cats[0].get("skeleton")
            keypoints_skeleton = [tuple(x) for x in keypoints_skeleton]

    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]

    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    id_map = {v: i for i, v in enumerate(cat_ids)}

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "area"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for img_dict, anno_dict_list in imgs_anns:
        image_id = img_dict["id"]
        record = DetectronDict(
            file_name=os.path.join(image_root, img_dict["file_name"]),
            height=img_dict["height"],
            width=img_dict["width"],
            image_id=img_dict["id"],
        )
        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm is not None and task == Task.INSTANCE_SEGMENTATION:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

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
        record.annotations = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(num_instances_without_valid_segmentation)
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )

    # Fill metadata information
    metadata.num_classes = len(thing_classes)
    metadata.thing_classes = thing_classes
    metadata.thing_dataset_id_to_contiguous_id = id_map
    metadata.keypoints = keypoints
    metadata.keypoints_skeleton = keypoints_skeleton
    if "color" in cats[0]:
        thing_colors = [c["color"] for c in sorted(cats, key=lambda x: x["id"])]
        metadata.thing_colors = thing_colors

    return dataset_dicts


def load_coco_panoptic_json(json_file, image_dir, gt_dir, metadata: DatasetMetadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format.
    """

    def _convert_category_id(
        segment_info,
        thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id,
    ):
        if segment_info["category_id"] in thing_dataset_id_to_contiguous_id:
            segment_info["category_id"] = thing_dataset_id_to_contiguous_id[segment_info["category_id"]]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = stuff_dataset_id_to_contiguous_id[segment_info["category_id"]]
            segment_info["isthing"] = False
        return segment_info

    with open(json_file) as f:
        json_info = json.load(f)

    categories = json_info["categories"]
    # All the classes are stuff, only a subset is thing
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(categories):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    metadata.thing_classes = [k["name"] for k in categories if k["isthing"] == 1]
    metadata.stuff_classes = [k["name"] for k in categories]
    metadata.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    metadata.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
    metadata.num_classes = len(categories)
    if "color" in categories[0]:
        metadata.thing_colors = [k["color"] for k in categories if k["isthing"] == 1]
        metadata.stuff_colors = [k["color"] for k in categories]

    images = dict()
    for info in json_info["images"]:
        images[info["id"]] = info["file_name"]

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]

        image_file = os.path.join(image_dir, images[image_id])
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [
            _convert_category_id(x, thing_dataset_id_to_contiguous_id, stuff_dataset_id_to_contiguous_id)
            for x in ann["segments_info"]
        ]
        ret.append(
            DetectronDict(
                file_name=image_file,
                image_id=image_id,
                pan_seg_file_name=label_file,
                segments_info=segments_info,
            )
        )
    return ret


def replace_path_prefix(path: str, new_prefix: str) -> str:
    parts = path.split("/")
    return "/".join([new_prefix] + parts[1:])


def remove_prefix(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[1:])
