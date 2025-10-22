"""
Default augmentation pipelines for Lightning DataModule.
Provides task-specific augmentations for training and validation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from focoos.ports import Task

from .transforms import LetterBox


def get_default_train_augmentations(task: Task, image_size: int) -> A.Compose:
    """
    Get default training augmentations based on task.

    Args:
        task: The task type (DETECTION, CLASSIFICATION, SEMSEG, etc.)
        image_size: Target image size for resizing

    Returns:
        Albumentations Compose object with training augmentations
    """
    if task == Task.CLASSIFICATION:
        return A.Compose(
            [  # type: ignore
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # LetterBox(height=image_size, width=image_size),
                A.Resize(height=image_size, width=image_size),
                ToTensorV2(),
            ]
        )

    elif task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
        # Mosaic viene applicato come prima augmentation nel pipeline
        transforms_list = [
            A.Mosaic(
                target_size=(image_size, image_size),
                metadata_key="mosaic_metadata",
                p=0.2,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.ColorJitter(p=0.2),
            LetterBox(height=image_size, width=image_size, p=1.0),
            # A.Resize(height=image_size, width=image_size),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
        )

    elif task == Task.SEMSEG:
        return A.Compose(
            [
                LetterBox(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    else:
        # Generic task augmentations
        return A.Compose(
            [
                LetterBox(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_default_val_augmentations(task: Task, image_size: int) -> A.Compose:
    """
    Get default validation augmentations based on task.

    Validation augmentations are minimal (only letterbox and normalize) to evaluate
    the model on data as close as possible to the original.

    Args:
        task: The task type (DETECTION, CLASSIFICATION, SEMSEG, etc.)
        image_size: Target image size for resizing

    Returns:
        Albumentations Compose object with validation augmentations
    """
    if task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
        return A.Compose(
            [
                LetterBox(height=image_size, width=image_size),
                # A.Resize(height=image_size, width=image_size),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="albumentations", label_fields=["labels"]),
        )
    else:
        return A.Compose(
            [
                LetterBox(height=image_size, width=image_size),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
