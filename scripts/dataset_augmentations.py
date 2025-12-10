# try to laod the downloaded dataset
import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.ports import DatasetSplitType, DetectronDict, Task
from focoos.structures import BoxMode

TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "valid"


def open_image(file_name) -> np.ndarray:
    image = np.array(Image.open(file_name))
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


class MosaicAugmentation:
    """
    Augmentation that combines 2 or 4 images in a simple grid layout.

    This augmentation creates a new image by combining images in a clean grid:
    - 2 images: Side by side (horizontal) or stacked (vertical)
    - 4 images: 2x2 grid

    Every pixel in the output image belongs to exactly one input image.
    """

    def __init__(self, dataset: DictDataset, output_size=(640, 640)):
        """
        Args:
            dataset: The dataset to sample images from
            output_size: Tuple (width, height) for the output mosaic image size
        """
        super().__init__()
        self.dataset = dataset
        self.output_size = output_size

    def _get_grid_layout(self, num_images):
        """Determine the grid layout based on number of images."""
        if num_images == 2:
            # Randomly choose horizontal or vertical split
            if random.random() < 0.5:
                return "horizontal"  # Side by side
            else:
                return "vertical"  # Stacked
        elif num_images == 4:
            return "2x2"  # 2x2 grid
        else:
            raise ValueError(f"Unsupported number of images: {num_images}. Use 2 or 4.")

    def _get_grid_regions(self, layout):
        """Calculate the exact regions for each image in the grid."""
        width, height = self.output_size

        if layout == "horizontal":
            # Two images side by side
            mid_x = width // 2
            return [
                (0, 0, mid_x, height),  # Left image
                (mid_x, 0, width, height),  # Right image
            ]
        elif layout == "vertical":
            # Two images stacked
            mid_y = height // 2
            return [
                (0, 0, width, mid_y),  # Top image
                (0, mid_y, width, height),  # Bottom image
            ]
        elif layout == "2x2":
            # Four images in 2x2 grid
            mid_x = width // 2
            mid_y = height // 2
            return [
                (0, 0, mid_x, mid_y),  # Top-left
                (mid_x, 0, width, mid_y),  # Top-right
                (0, mid_y, mid_x, height),  # Bottom-left
                (mid_x, mid_y, width, height),  # Bottom-right
            ]
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def transform(self, dataset_dict):
        """
        Apply mosaic augmentation to create a grid of images.

        Args:
            dataset_dict: Dictionary containing image information and annotations

        Returns:
            mosaic_dataset_dict: Updated dataset dict with mosaic image and combined annotations
        """
        # Determine number of images (2 or 4)
        num_images = random.choice([2, 4])

        # Sample random images from dataset
        source_indices = [random.randint(0, len(self.dataset) - 1) for _ in range(num_images - 1)]
        source_indices.insert(0, random.randint(0, len(self.dataset) - 1))  # Include original image

        # Load all images
        images = []
        all_annotations = []

        for i, idx in enumerate(source_indices):
            if i == 0:
                # Use the original image
                image = open_image(dataset_dict["file_name"])
                annotations = dataset_dict.get("annotations", []).copy()
            else:
                # Sample from dataset
                sample = self.dataset[idx]
                image = open_image(sample["file_name"])
                annotations = sample.get("annotations", []).copy()

            images.append(image)
            all_annotations.append(annotations)

        # Determine grid layout
        layout = self._get_grid_layout(num_images)
        regions = self._get_grid_regions(layout)

        # Create output mosaic image
        mosaic_image = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        mosaic_annotations = []

        # Process each image and place it in the grid
        for i, (image, annotations, region) in enumerate(zip(images, all_annotations, regions)):
            x1, y1, x2, y2 = region
            region_width = x2 - x1
            region_height = y2 - y1

            # Resize image to exactly fit the region (no padding, no gaps)
            img_h, img_w = image.shape[:2]

            # Resize to fit the exact region dimensions
            resized_image = np.array(Image.fromarray(image).resize((region_width, region_height)))

            # Place the image in the exact region
            mosaic_image[y1:y2, x1:x2] = resized_image

            # Update annotations for this image
            for ann in annotations:
                if "bbox" in ann:
                    # Convert bbox to absolute coordinates
                    box = BoxMode.convert(
                        ann["bbox"],
                        ann.get("bbox_mode", BoxMode.XYWH_ABS),
                        BoxMode.XYXY_ABS,
                    )
                    assert ann.get("category_id") is not None, "category_id is None"

                    # Scale the bbox to fit the new region
                    x1_orig, y1_orig, x2_orig, y2_orig = box

                    # Calculate scale factors
                    scale_x = region_width / img_w
                    scale_y = region_height / img_h

                    # Scale and translate the bbox
                    new_x1 = int(x1_orig * scale_x) + x1
                    new_y1 = int(y1_orig * scale_y) + y1
                    new_x2 = int(x2_orig * scale_x) + x1
                    new_y2 = int(y2_orig * scale_y) + y1

                    # Check if bbox is within the output image bounds
                    if (
                        new_x1 >= 0
                        and new_y1 >= 0
                        and new_x2 <= self.output_size[0]
                        and new_y2 <= self.output_size[1]
                        and new_x2 > new_x1
                        and new_y2 > new_y1
                    ):
                        # Convert back to XYWH format
                        new_w_bbox = new_x2 - new_x1
                        new_h_bbox = new_y2 - new_y1
                        new_area = new_w_bbox * new_h_bbox

                        mosaic_annotations.append(
                            {
                                "bbox": [new_x1, new_y1, new_w_bbox, new_h_bbox],
                                "bbox_mode": BoxMode.XYWH_ABS,
                                "category_id": ann.get("category_id"),
                                "area": new_area,
                                "iscrowd": ann.get("iscrowd", 0),
                            }
                        )

        # Create the mosaic dataset dict
        mosaic_dataset_dict = dataset_dict.copy()
        mosaic_dataset_dict["image"] = mosaic_image
        mosaic_dataset_dict["annotations"] = mosaic_annotations
        mosaic_dataset_dict["width"] = self.output_size[0]
        mosaic_dataset_dict["height"] = self.output_size[1]

        return mosaic_dataset_dict


class CopyPasteAugmentation:
    """
    Augmentation that enhances an image by pasting objects from other images onto it.
    """

    def __init__(self, dataset: DictDataset, scale_range=(0.3, 0.7), num_objects=3, blend_factor=1.0):
        """
        Args:
            dataset: The dataset to sample images from
            scale_range: Range of scaling factors for the pasted objects
            num_objects: Number of objects to paste onto the original image
            blend_factor: Factor controlling the blending of pasted objects
        """
        super().__init__()
        self.dataset = dataset
        self.scale_range = scale_range
        self.num_objects = num_objects
        self.blend_factor = blend_factor

    def transform(self, dataset_dict):
        """
        Apply copy paste augmentation to an image.

        Args:
            dataset_dict: Dictionary containing image information and annotations

        Returns:
            copy_paste_dataset_dict: Updated dataset dict with augmented image and annotations
        """
        # Load the original image
        image = open_image(dataset_dict["file_name"])
        # Convert grayscale to RGB if needed
        h, w = image.shape[:2]
        copy_paste_image = image.copy()

        # Create a dataset dict for the mosaic image
        copy_paste_dataset_dict = dataset_dict.copy()
        original_annotations = copy_paste_dataset_dict.get("annotations", []).copy()
        copy_paste_dataset_dict["annotations"] = original_annotations

        # Create a mask to track occupied regions
        occupied_mask = np.zeros((h, w), dtype=bool)

        # Calculate maximum allowed area (10% of original image)
        max_allowed_area = 0.1 * h * w

        # Sample random images to get objects from
        source_indices = [random.randint(0, len(self.dataset) - 1) for _ in range(self.num_objects)]

        for source_idx in source_indices:
            # Get a random image from dataset
            sample = self.dataset[source_idx]
            source_image = open_image(sample["file_name"])

            # Get bounding boxes if available
            if "annotations" in sample and len(sample["annotations"]) > 0:
                source_boxes = []
                category_ids = []
                areas = []
                iscrowds = []

                for ann in sample["annotations"]:
                    if "bbox" in ann:
                        box = BoxMode.convert(
                            ann["bbox"],
                            ann.get("bbox_mode", BoxMode.XYWH_ABS),
                            BoxMode.XYXY_ABS,
                        )
                        source_boxes.append(box)
                        category_ids.append(ann.get("category_id"))
                        areas.append(ann.get("area", 0))
                        iscrowds.append(ann.get("iscrowd", 0))
                source_boxes = np.array(source_boxes)
            else:
                continue  # Skip if no boxes

            if len(source_boxes) == 0:
                continue

            # Select a random box
            box_idx = random.randint(0, len(source_boxes) - 1)
            box = source_boxes[box_idx]
            category_id = category_ids[box_idx]
            # area = areas[box_idx]
            iscrowd = iscrowds[box_idx]

            # Get the object region
            x1, y1, x2, y2 = map(int, box)
            box_w, box_h = x2 - x1, y2 - y1

            # Add margin around the box
            margin = 0.1
            crop_x1 = max(0, int(x1 - margin * box_w))
            crop_y1 = max(0, int(y1 - margin * box_h))
            crop_x2 = min(source_image.shape[1], int(x2 + margin * box_w))
            crop_y2 = min(source_image.shape[0], int(y2 + margin * box_h))

            # Crop the object region
            crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
            cropped_image = source_image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Scale the cropped image
            scale_factor = random.uniform(*self.scale_range)
            new_w, new_h = int(crop_w * scale_factor), int(crop_h * scale_factor)

            # Skip if too small
            if new_w < 10 or new_h < 10:
                continue

            # Skip if area exceeds maximum allowed area (10% of original image)
            if new_w * new_h > max_allowed_area:
                continue

            scaled_image = np.array(Image.fromarray(cropped_image).resize((new_w, new_h)))

            # Find a place to paste (avoid edges)
            max_x = w - new_w
            max_y = h - new_h
            if max_x <= 0 or max_y <= 0:
                continue

            # Try to find a non-overlapping position (max 10 attempts)
            found_valid_position = False
            for _ in range(10):
                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)

                # Check if the region overlaps with existing objects
                region_mask = occupied_mask[paste_y : paste_y + new_h, paste_x : paste_x + new_w]
                if region_mask.size > 0 and not np.any(region_mask):
                    found_valid_position = True
                    break

            # Skip if we couldn't find a non-overlapping position
            if not found_valid_position:
                continue

            # Mark this region as occupied
            occupied_mask[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = True

            # Create alpha mask for smoother blending
            alpha = self.blend_factor  # Blend factor

            # Paste the scaled image with alpha blending
            roi = copy_paste_image[paste_y : paste_y + new_h, paste_x : paste_x + new_w]
            if roi.shape[:2] == scaled_image.shape[:2]:  # Ensure shapes match
                copy_paste_image[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = (
                    alpha * scaled_image + (1 - alpha) * roi
                ).astype(np.uint8)

            # Calculate new area
            new_area = new_w * new_h

            # Add the new box to the annotations in XYWH_ABS format
            copy_paste_dataset_dict["annotations"].append(
                {
                    "bbox": [paste_x, paste_y, new_w, new_h],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": category_id,
                    "area": new_area,
                    "iscrowd": iscrowd,
                }
            )

        # Update the image in the dataset dict
        copy_paste_dataset_dict["image"] = copy_paste_image

        return copy_paste_dataset_dict


# Example usage in a notebook:
def apply_mosaic_augmentation(dataset, num_samples=5, output_size=(640, 640)):
    """Apply mosaic augmentation to a few samples and display results"""
    mosaic_aug = MosaicAugmentation(dataset, output_size=output_size)

    results = []
    for i in range(num_samples):
        sample_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]

        # Apply mosaic augmentation
        new_dict = mosaic_aug.transform(sample)

        results.append((new_dict))

    return results


def apply_copypaste_augmentation(dataset, num_samples=5, num_objects=5):
    """Apply copy-paste augmentation to a few samples and display results"""
    copy_paste_aug = CopyPasteAugmentation(dataset, num_objects=num_objects)

    results = []
    for i in range(num_samples):
        sample_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]

        # Apply copy-paste augmentation
        new_dict = copy_paste_aug.transform(sample)

        results.append((new_dict))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create augmented dataset with increased size")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/coco", help="Path to the original dataset")
    parser.add_argument(
        "--copy_paste_weight",
        type=int,
        default=5,
        help="Weight of copy-paste augmentation (10 is same as no augmentation)",
    )
    parser.add_argument(
        "--mosaic_weight",
        type=int,
        default=5,
        help="Weight of mosaic augmentation (10 is same as no augmentation)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Target number of images in the resulting dataset. If not specified, will use original size",
    )
    parser.add_argument(
        "--augmentation_ratio",
        type=float,
        default=1.0,
        help="Ratio of augmented images to original images (e.g., 1.0 means equal number of original images)",
    )
    parser.add_argument(
        "--copy_paste_num_objects", type=int, default=10, help="Number of objects to paste in copy-paste augmentation"
    )
    parser.add_argument(
        "--mosaic_output_size", type=str, default="640,640", help="Comma-separated width,height for mosaic output"
    )
    parser.add_argument(
        "--copy_val_dataset",
        action="store_true",
        help="Copy validation dataset alongside the augmented training split",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_augmented",
        help="Suffix for output dataset directory (default: based on augmentation type)",
    )
    args = parser.parse_args()

    # Load original datasets
    print("Loading original datasets...")
    data = DictDataset.from_roboflow_coco(
        ds_dir=args.dataset_dir + "/" + TRAIN_SPLIT_NAME, task=Task.DETECTION, split_type=DatasetSplitType.TRAIN
    )
    val_data = DictDataset.from_roboflow_coco(
        ds_dir=args.dataset_dir + "/" + VAL_SPLIT_NAME, task=Task.DETECTION, split_type=DatasetSplitType.VAL
    )

    original_size = len(data)
    print(f"Original training dataset size: {original_size}")

    # Calculate target size
    if args.target_size is None:
        target_size = int(original_size * (args.augmentation_ratio))
    else:
        target_size = args.target_size

    augmented_size = target_size
    print(f"Target dataset size: {target_size}")

    # Parse mosaic parameters
    mosaic_output_size = tuple(map(int, args.mosaic_output_size.split(",")))

    # Choose augmentation type
    copypaste_aug = CopyPasteAugmentation(data, num_objects=args.copy_paste_num_objects)
    mosaic_aug = MosaicAugmentation(data, output_size=mosaic_output_size)

    new_dataset_root = args.dataset_dir + args.output_suffix
    new_train_root = new_dataset_root + "/" + TRAIN_SPLIT_NAME

    os.makedirs(new_dataset_root, exist_ok=True)
    os.makedirs(new_dataset_root + "/" + TRAIN_SPLIT_NAME, exist_ok=True)

    augmentation_weights = args.copy_paste_weight + args.mosaic_weight + 10

    # Generate augmented images
    new_dataset = []

    if augmented_size > 0:
        print(f"Generating {augmented_size} augmented images...")

        # Calculate how many times we need to go through the original dataset
        iterations_needed = (augmented_size + original_size - 1) // original_size

        for iteration in range(iterations_needed):
            remaining_augmented = augmented_size - len(new_dataset)
            if remaining_augmented <= 0:
                break

            print(f"Augmentation iteration {iteration + 1}/{iterations_needed}")

            for dic in tqdm(data, desc=f"Generating augmented images (iter {iteration + 1})"):
                if len(new_dataset) >= augmented_size:
                    break

                try:
                    # Apply augmentation
                    augmentation = random.random() * augmentation_weights
                    if augmentation < args.copy_paste_weight:
                        aug = copypaste_aug
                    elif augmentation < (args.copy_paste_weight + args.mosaic_weight):
                        aug = mosaic_aug
                    else:
                        aug = None

                    if aug is not None:
                        augmented_dict = aug.transform(dic)
                    else:
                        augmented_dict = dic
                        augmented_dict["image"] = open_image(dic["file_name"])

                    # Generate unique filename
                    original_file_name = dic["file_name"].split("/")[-1]
                    name, ext = os.path.splitext(original_file_name)
                    aug_file_name = f"{name}_aug_{len(new_dataset) + 1}{ext}"
                    new_file_name = os.path.join(new_train_root, aug_file_name)

                    # Save augmented image
                    Image.fromarray(augmented_dict["image"]).save(new_file_name)

                    # Update the dataset dict
                    augmented_dict["file_name"] = new_file_name
                    del augmented_dict["image"]  # Remove image data to save memory
                    new_dataset.append(augmented_dict)

                except Exception as e:
                    print(f"Error generating augmented image ({dic['file_name']}): {e}")
                    # Skip this augmentation if it fails
                    continue

    # Create the final dataset
    print("Creating final dataset...")
    final_dataset = DictDataset(
        [DetectronDict(**d) for d in new_dataset],
        metadata=data.metadata,
        task=Task.DETECTION,
        split_type=DatasetSplitType.TRAIN,
    )

    print(f"Final dataset size: {len(final_dataset)}")

    # Save the dataset
    final_dataset.save(output_dir=new_dataset_root + "/" + TRAIN_SPLIT_NAME)

    # Copy validation dataset if requested
    if args.copy_val_dataset:
        print("Copying validation dataset...")
        shutil.copytree(args.dataset_dir + "/" + VAL_SPLIT_NAME, new_dataset_root + "/" + VAL_SPLIT_NAME)

    print(f"Augmented dataset saved to: {new_dataset_root}")
    print("Dataset augmentation completed successfully!")
