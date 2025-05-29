import base64
import csv
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

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import Task
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def base64_to_numpy(base64_string):
    image_data = zlib.decompress(base64.b64decode(base64_string))
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.bool)
    return image


def get_classes(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    classes = data["classes"]
    return {cls["title"]: i for i, cls in enumerate(classes)}


def convert_json_to_png(json_file: str, class_to_id):
    with open(json_file, "r") as f:
        data = json.load(f)

    output_png = np.zeros((data["size"]["height"], data["size"]["width"]), dtype=np.uint8) - 1

    for annotation in data["objects"]:
        class_name = annotation["classTitle"]
        class_id = class_to_id[class_name]
        if annotation["geometryType"] == "bitmap":
            origin = np.array(annotation["bitmap"]["origin"])
            mask_b64 = annotation["bitmap"]["data"]
            mask = base64_to_numpy(mask_b64)

            output_png[origin[1] : origin[1] + mask.shape[0], origin[0] : origin[0] + mask.shape[1]][mask] = class_id
        else:
            print(f"Warning: Unsupported geometry type: {annotation['geometryType']}")

    return output_png


def convert_dataset(dataset_root, remove_json=False):
    """ "
    Convert a Supervisely dataset annotations to the mask format.
    Given the json, it stores the class id for each pixel in a png file with the same name as the json file but with .png extension.
    The dataset is expected to be in the following structure:
    dataset_root/
        meta.json
        {train/val/test/any}/
            {image_name}/
                file1.jpg
                file2.jpg
            {ann_name}/
                file1.json
                file2.json
    """
    class_to_id = get_classes(os.path.join(dataset_root, "meta.json"))
    for folder in os.listdir(dataset_root):
        if os.path.isfile(os.path.join(dataset_root, folder)):
            continue
        logger.info(f"Processing folder {folder}")
        for subfolder in os.listdir(os.path.join(dataset_root, folder)):
            if os.path.isfile(os.path.join(dataset_root, folder, subfolder)):
                continue
            for file in os.listdir(os.path.join(dataset_root, folder, subfolder)):
                if file.endswith(".json"):
                    png_output = convert_json_to_png(os.path.join(dataset_root, folder, subfolder, file), class_to_id)
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
