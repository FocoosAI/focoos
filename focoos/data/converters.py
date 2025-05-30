import base64
import concurrent.futures
import csv
import datetime
import json
import os
import random
import shutil
import zlib
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.transforms.resize_short_length import resize_shortest_length
from focoos.ports import DatasetMetadata, Task
from focoos.utils.logger import get_logger
from focoos.utils.system import list_files_with_extensions

logger = get_logger(__name__)


def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def base64_to_numpy(base64_string):
    image_data = zlib.decompress(base64.b64decode(base64_string))
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.bool)
    return image


def get_classes(json_file: str, use_background: bool = False, ignore_classes: List[str] = []):
    with open(json_file, "r") as f:
        data = json.load(f)
    classes = data["classes"]

    new_classes = {"background": 0} if use_background else {}
    for idx, cls in enumerate(classes):
        if cls["title"] not in ignore_classes:
            new_classes[cls["title"]] = idx + 1 if use_background else idx
    return new_classes


def convert_json_to_png(json_file: str, class_to_id, use_background: bool = False, ignore_classes: List[str] = []):
    with open(json_file, "r") as f:
        data = json.load(f)

    if use_background:
        output_png = np.zeros((data["size"]["height"], data["size"]["width"]), dtype=np.uint8)
    else:
        output_png = np.zeros((data["size"]["height"], data["size"]["width"]), dtype=np.uint8) - 1

    for annotation in data["objects"]:
        class_name = annotation["classTitle"]
        class_id = class_to_id[class_name] if use_background else class_to_id[class_name] + 1
        if class_name in ignore_classes:
            class_id = 255
        if annotation["geometryType"] == "bitmap":
            origin = np.array(annotation["bitmap"]["origin"])
            mask_b64 = annotation["bitmap"]["data"]
            mask = base64_to_numpy(mask_b64)

            output_png[origin[1] : origin[1] + mask.shape[0], origin[0] : origin[0] + mask.shape[1]][mask] = class_id
        else:
            print(f"Warning: Unsupported geometry type: {annotation['geometryType']}")

    return output_png


def convert_supervisely_dataset_to_png(
    dataset_root, remove_json=False, use_background=False, ignore_classes=[], ignore_folders=[]
):
    """
    Convert Supervisely dataset annotations to mask format.

    This function processes Supervisely-formatted JSON annotations and converts them into PNG mask files.
    Each pixel in the output mask is assigned a class ID corresponding to its annotation.

    Args:
        dataset_root (str): Path to the root directory of the dataset.
        remove_json (bool, optional): Whether to remove the original JSON files after conversion. Defaults to False.
        use_background (bool, optional): If True, assigns class ID 0 to non-annotated pixels and shifts other class IDs by 1. Defaults to False.
        ignore_classes (List[str], optional): List of class names to ignore during conversion. Defaults to [].

    Expected Directory Structure:
        dataset_root/
            meta.json
            {train/val/test/any}/
                {image_name}/
                    file1.jpg
                    file2.jpg
                {ann_name}/
                    file1.json
                    file2.json

    Returns:
        None: Creates PNG mask files in the same directory as the input JSON files.
    """
    class_to_id = get_classes(os.path.join(dataset_root, "meta.json"))
    for folder in os.listdir(dataset_root):
        if os.path.isfile(os.path.join(dataset_root, folder)) or folder in ignore_folders:
            continue
        logger.info(f"Processing folder {folder}")
        for subfolder in os.listdir(os.path.join(dataset_root, folder)):
            if os.path.isfile(os.path.join(dataset_root, folder, subfolder)) or subfolder in ignore_folders:
                continue
            for file in os.listdir(os.path.join(dataset_root, folder, subfolder)):
                if file.endswith(".json"):
                    png_output = convert_json_to_png(
                        os.path.join(dataset_root, folder, subfolder, file),
                        class_to_id,
                        use_background,
                        ignore_classes,
                    )
                    Image.fromarray(png_output).save(
                        os.path.join(dataset_root, folder, subfolder, file.replace(".jpg.json", ".png"))
                    )
                    if remove_json:
                        os.remove(os.path.join(dataset_root, folder, subfolder, file))


def create_segmentation_json(
    root_dir: str,
    image_folder: str,
    mask_folder: str,
    classes: List[str],
    output_file: str = "annotations.json",
    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
    mask_suffix: str = ".png",
):
    """
    Create a json file for a segmentation dataset

    The classes should be a list of strings following the id in the png file.

    The dataset is expected to be in the following structure:
    root_dir/  -> This is likely the split folder (train, val, test, etc.)
        {image_folder}/
            {image_name}.{image_extension}
        {mask_folder}/
            {image_name}.{mask_suffix}

    It will create a json file with the following structure that will be accepted as the DictDataset.from_segmentation input.
    {
        "images": [
            {
                "id": int,
                "file_name": str,
                "height": int,
                "width": int,
            },
            ...
        ],
        "annotations": [
            {
                "image_id": int,
                "file_name": str,
            },
            ...
        ],
        "categories": [
            {
                "id": int,
                "name": str,
                "color": [int, int, int],
                "is_thing": bool,
            },
            ...
        ]
    }
    """
    images = []
    annotations = []
    categories = []

    # Create a mapping from class name to class ID
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    for class_name in classes:
        categories.append(
            {
                "id": class_to_id[class_name],
                "name": class_name,
                "color": get_random_color(),
                "is_thing": True,
            }
        )

    for idx, image in enumerate(os.listdir(os.path.join(root_dir, image_folder))):
        if Path(image).suffix not in image_extensions:
            continue
        mask_path = os.path.join(mask_folder, Path(image).stem + mask_suffix)

        if not os.path.exists(os.path.join(root_dir, mask_path)):
            print(f"Warning: Mask file {mask_path} does not exist")
            continue

        image_path = os.path.join(root_dir, image_folder, image)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            print(f"Warning: Image file {image_path} is not a valid image")
            continue

        images.append(
            {
                "id": idx,
                "file_name": os.path.join(image_folder, image),
                "height": height,
                "width": width,
            }
        )

        annotations.append(
            {
                "image_id": idx,
                "file_name": mask_path,
            }
        )

    json_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(os.path.join(root_dir, output_file), "w") as f:
        json.dump(json_data, f)


def convert_to_mask_format(dict_dataset: DictDataset, new_data_dir: str):
    """
    Convert a DictDataset to the Mask Format of roboflow.
    The output directory will be structured as follows:
    new_data_dir/ -> This is likely the split folder (train, val, test, etc.)
        _classes.csv
        {image_name}.{image_extension}
        {image_name}_mask.png

    The classes.csv file will be created with the following structure:
    Pixel Value,Class
    0,unlabeled

    """
    assert dict_dataset.metadata.task == Task.SEMSEG, "Error, not a SEMSEG dataset"
    os.makedirs(new_data_dir, exist_ok=True)

    # Create classes.csv file
    classes_file = os.path.join(new_data_dir, "_classes.csv")
    with open(classes_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pixel Value", "Class"])  # Write header
        for class_id, class_name in enumerate(dict_dataset.metadata.classes):
            writer.writerow([class_id, class_name])

    for diz in dict_dataset:
        image = diz["file_name"]
        mask = diz["sem_seg_file_name"]
        new_img_path = Path(image).name
        new_mask_path = new_img_path[:-4] + "_mask.png"
        shutil.copy(image, os.path.join(new_data_dir, new_img_path))
        shutil.copy(mask, os.path.join(new_data_dir, new_mask_path))


def clone_resize_shortest_length(dataset: DictDataset, new_dir: str, new_shortest_length: int = 1024, max_size=2048):
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
    orig_meta = dataset.metadata

    for data in dataset.dicts:  # type: ignore
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
            mask_file,  # type: ignore
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


def get_annotation_dict_from_json_file(json_file: str, image_id, start_annotation_id, class_to_id):
    with open(json_file, "r") as f:
        data = json.load(f)

    annotations = []
    annotation_id = start_annotation_id
    for annotation in data["objects"]:
        if annotation["geometryType"] == "rectangle":
            if annotation["classTitle"] not in class_to_id:
                logger.info(f"Skipping annotation {annotation['classTitle']} because it is ignored")
                continue
            class_id = class_to_id[annotation["classTitle"]] + 1  # in COCO the 0 is ignored
            bbox = annotation["points"]["exterior"]
            bbox = np.array(
                [bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]], dtype=np.float32
            )  # Convert to xyxy format
            area = bbox[2] * bbox[3]  # Calculate area from xyxy coordinates
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": bbox.tolist(),
                    "area": int(area),
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        else:
            raise ValueError(f"Unsupported geometry type: {annotation['geometryType']}")

    return annotations


def convert_datasetninja_to_mask_dataset(
    dataset_root: str,
    dataset_name: str,
    new_name: str,
    image_folder: str,
    mask_folder: str,
    ignore_folders: List[str] = [],
    use_background: bool = True,
    ignore_classes: List[str] = [],
    train_split_name: str = "train",
    val_split_name: str = "val",
    remove_json: bool = False,
):
    """Convert a DatasetNinja dataset to a mask-based segmentation dataset format.

    This function performs a multi-step conversion process:
    1. Converts DatasetNinja JSON annotations to PNG masks
    2. Creates segmentation JSON files for train and validation splits
    3. Converts the dataset to a mask-based format compatible with RoboflowMask format

    Args:
        dataset_root (str): Root directory containing the dataset
        dataset_name (str): Name of the source DatasetNinja dataset folder
        new_name (str): Name for the converted dataset folder
        image_folder (str): Name of the folder containing images
        mask_folder (str): Name of the folder containing masks
        ignore_folders (List[str], optional): List of folders to ignore during conversion. Defaults to [].
        use_background (bool, optional): Whether to include background class. Defaults to True.
        ignore_classes (List[str], optional): List of classes to ignore. Defaults to [].
        train_split_name (str, optional): Name of the training split folder. Defaults to "train".
        val_split_name (str, optional): Name of the validation split folder. Defaults to "test".
        remove_json (bool, optional): Whether to remove original JSON files after conversion. Defaults to False.

    Expected Directory Structure:
        dataset_root/
            dataset_name/
                meta.json
                {train_split_name}/
                    {image_folder}/
                        image1.jpg
                        image2.jpg
                    {mask_folder}/
                        image1.json
                        image2.json
                {val_split_name}/
                    {image_folder}/
                        image1.jpg
                        image2.jpg
                    {mask_folder}/
                        image1.json
                        image2.json

    Output Dataset Structure:
        dataset_root/
            new_dataset_name/
                train/
                    _classes.csv
                    image1.jpg
                    image1_mask.png
                val/
                    _classes.csv
                    image1.jpg
                    image1_mask.png

    Returns:
        None: The converted dataset is saved to the specified output directory.
    """
    dataset_path = os.path.join(dataset_root, dataset_name)
    new_dataset_path = os.path.join(dataset_root, new_name)

    logger.info(f"Converting {dataset_name} from DatasetNinja Json to PNG")
    convert_supervisely_dataset_to_png(
        dataset_root=dataset_path,
        use_background=use_background,
        ignore_classes=ignore_classes,
        ignore_folders=ignore_folders,
        remove_json=remove_json,
    )

    classes = get_classes(
        os.path.join(dataset_path, "meta.json"), use_background=use_background, ignore_classes=ignore_classes
    )
    logger.info(f"Classes: {classes}")

    for split in [train_split_name, val_split_name]:
        logger.info(f"Creating segmentation json for {split}")
        create_segmentation_json(
            root_dir=os.path.join(dataset_path, split),
            image_folder=image_folder,
            mask_folder=mask_folder,
            classes=list(classes.keys()),
        )

    task = Task.SEMSEG
    train_dataset = DictDataset.from_segmentation(ds_dir=os.path.join(dataset_path, train_split_name), task=task)
    logger.info(f"Train dataset: {train_dataset}")

    val_dataset = DictDataset.from_segmentation(ds_dir=os.path.join(dataset_path, val_split_name), task=task)
    logger.info(f"Val dataset: {val_dataset}")

    for split in [(train_dataset, "train"), (val_dataset, "val")]:
        logger.info(f"Converting {split[1]} dataset to mask format into")
        convert_to_mask_format(dict_dataset=split[0], new_data_dir=os.path.join(new_dataset_path, split[1]))


def convert_supervisely_dataset_to_coco(
    dataset_root: str,
    dataset_name: str,
    new_name: str,
    image_folder: str,
    mask_folder: str,
    ignore_classes: List[str] = [],
    train_split_name: str = "train",
    val_split_name: str = "val",
    remove_json: bool = False,
):
    """
    Convert Supervisely dataset annotations to COCO format.

    This function processes Supervisely-formatted JSON annotations and converts them into COCO format.
    The conversion preserves image metadata, annotations, and class information while adapting to COCO's
    specific structure and requirements.

    Args:
        dataset_root (str): Path to the root directory of the dataset.
        dataset_name (str): Name of the dataset directory containing the Supervisely annotations.
        new_name (str): Name for the converted COCO dataset.
        image_folder (str): Name of the folder containing the images.
        mask_folder (str): Name of the folder containing the annotation files.
        ignore_classes (List[str], optional): List of class names to ignore during conversion. Defaults to [].
        train_split_name (str, optional): Name of the training split folder. Defaults to "train".
        val_split_name (str, optional): Name of the validation split folder. Defaults to "val".
        remove_json (bool, optional): Whether to remove the original JSON files after conversion. Defaults to False.

    Expected Directory Structure:
        dataset_root/
            dataset_name/
                meta.json
                {train_split_name}/
                    {image_folder}/
                        image1.jpg
                        image2.jpg
                    {mask_folder}/
                        image1.json
                        image2.json
                {val_split_name}/
                    {image_folder}/
                        image1.jpg
                        image2.jpg
                    {mask_folder}/
                        image1.json
                        image2.json

    Output Dataset Structure:
        dataset_root/
            new_dataset_name/
                train/
                    _annotations.coco.json
                    image1.jpg
                val/
                    _annotations.coco.json
                    image1.jpg

    Returns:
        None: Creates a new directory with COCO-formatted annotations and copies images to the new structure.
    """
    dataset_path = os.path.join(dataset_root, dataset_name)
    new_dataset_path = os.path.join(dataset_root, dataset_name + "_coco")

    class_to_id = get_classes(os.path.join(dataset_path, "meta.json"), ignore_classes=ignore_classes)

    info = {
        "year": "2025",
        "description": "Converted with Focoos",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    }
    categories = []

    for class_name, class_id in class_to_id.items():
        categories.append(
            {
                "id": class_id + 1,  # in COCO the 0 is ignored
                "name": class_name,
                "supercategory": "superclass",
            }
        )

    for split in [(train_split_name, "train"), (val_split_name, "val")]:
        images = []
        annotations = []

        os.makedirs(os.path.join(new_dataset_path, split[1]), exist_ok=True)

        logger.info(f"Processing folder {split[0]}")
        for file in tqdm(os.listdir(os.path.join(dataset_path, split[0], image_folder))):
            try:
                image = Image.open(os.path.join(dataset_path, split[0], image_folder, file))
                width, height = image.size
            except Exception:
                logger.warning(f"Image {file} is not a valid image")
                continue

            images.append(
                {
                    "id": len(images),
                    "file_name": file,
                    "height": height,
                    "width": width,
                }
            )
            shutil.copy(
                os.path.join(dataset_path, split[0], image_folder, file), os.path.join(new_dataset_path, split[1], file)
            )

            image_annotations = get_annotation_dict_from_json_file(
                os.path.join(dataset_path, split[0], mask_folder, file + ".json"),
                len(images) - 1,
                len(annotations),
                class_to_id,
            )
            if remove_json:
                os.remove(os.path.join(dataset_path, split[0], mask_folder, file, ".json"))

            annotations.extend(image_annotations)

        coco_json = {
            "info": info,
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        with open(os.path.join(new_dataset_path, split[1], "_annotations.coco.json"), "w") as f:
            json.dump(coco_json, f)
