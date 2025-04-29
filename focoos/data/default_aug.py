import copy
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from focoos.data.transforms import augmentation as A
from focoos.data.transforms import transform as T
from focoos.ports import Task
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetAugmentations:
    """
    Configuration class for dataset augmentations.

    This class defines parameters for various image transformations used in training and validation
    pipelines for computer vision tasks. It provides a comprehensive set of options for both
    color and geometric augmentations.

    Attributes:
        resolution (int): Target image size for resizing operations.
            Range [256, 1024]. Default: 640.
        ==
        color_augmentation (float): Strenght of color augmentations.
            Range [0,1]. Default: 0.0.
        ==
        horizontal_flip (float): Probability of applying horizontal flip.
            Range [0,1]. Default: 0.0.
        vertical_flip (float): Probability of applying vertical flip.
            Range [0,1]. Default: 0.0.
        zoom_out (float): Probability of applying RandomZoomOut.
            Range [0,1]. Default: 0.0.
        zoom_out_side (float): Zoom out side range.
            Range [1,5]. Default: 4.0.
        rotation (float): Probability of applying RandomRotation. 1 equals +/-180 degrees.
            Range [0,1]. Default: 0.0.
        ==
        square (bool): Whether to Square the image.
            Default: False.
        aspect_ratio (float): Aspect ratio for resizing (actual scale range is (2 ** -aspect_ratio, 2 ** aspect_ratio).
            Range [0,1]. Default: 0.0.
        scale_ratio (Optional[float]): scale factor for resizing (actual scale range is (2 ** -scale_ratio, 2 ** scale_ratio).
            Range [0,1]. Default: None.
        max_size (Optional[int]): Maximum allowed dimension after resizing.
            Range [256, sys.maxsize]. Default: sys.maxsize.
        ==
        crop (bool): Whether to apply RandomCrop.
            Default: False.
        crop_size_min (Optional[int]): Minimum crop size for RandomCrop.
            Range [256, 1024]. Default: None.
        crop_size_max (Optional[int]): Maximum crop size for RandomCrop.
            Range [256, 1024]. Default: None.
    """

    # Resolution for resizing
    resolution: int = 640

    # Color augmentation parameters
    color_augmentation: float = 0.0
    color_base_brightness: int = 32
    color_base_saturation: float = 0.5
    color_base_contrast: float = 0.5
    color_base_hue: float = 18
    # blur: float = 0.0
    # noise: float = 0.0

    # Geometric augmentation
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0
    zoom_out: float = 0.0
    zoom_out_side: float = 4.0
    rotation: float = 0.0
    aspect_ratio: float = 0.0

    ## Rescaling
    square: float = 0.0
    scale_ratio: float = 0.0
    max_size: int = 4096

    # Cropping
    crop: bool = False
    crop_size: Optional[int] = None

    # TODO: Add more augmentations like:
    # - GaussianBlur
    # - RandomNoise
    # - RandomResizedCrop

    def override(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        for key, value in args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

    def get_augmentations(self, img_format="RGB", task: Optional[Task] = None) -> List[T.Transform]:
        """Generate augmentation pipeline based on configuration."""
        augs = []
        self.max_size = self.max_size if self.max_size else sys.maxsize

        ### Add color augmentation if configured
        if self.color_augmentation > 0:
            brightness_delta = int(self.color_base_brightness * self.color_augmentation)
            contrast_delta = self.color_base_contrast * self.color_augmentation
            saturation_delta = self.color_base_saturation * self.color_augmentation
            hue_delta = int(self.color_base_hue * self.color_augmentation)
            augs.append(
                T.ColorAugSSDTransform(
                    img_format=img_format,
                    brightness_delta=brightness_delta,
                    contrast_low=(1 - contrast_delta),
                    contrast_high=(1 + contrast_delta),
                    saturation_low=(1 - saturation_delta),
                    saturation_high=(1 + saturation_delta),
                    hue_delta=hue_delta,
                ),
            )

        ### Add geometric augmentations
        # Add flipping augmentations if configured
        if self.horizontal_flip > 0:
            augs.append(A.RandomFlip(prob=self.horizontal_flip, horizontal=True))
        if self.vertical_flip > 0:
            augs.append(A.RandomFlip(prob=self.vertical_flip, horizontal=False, vertical=True))

        # Add zoom out augmentations if configured
        if self.zoom_out > 0.0:
            seg_pad_value = 255 if task == Task.SEMSEG else 0
            augs.append(
                A.RandomApply(
                    A.RandomZoomOut(side_range=(1.0, self.zoom_out_side), pad_value=0, seg_pad_value=seg_pad_value),
                    prob=self.zoom_out,
                )
            )

        ### Add AspectRatio augmentations based on configuration
        if self.square > 0.0:
            augs.append(A.RandomApply(A.Resize(shape=(self.resolution, self.resolution)), prob=self.square))
        elif self.aspect_ratio > 0.0:
            augs.append(A.RandomAspectRatio(aspect_ratio=self.aspect_ratio))

        ### Add Resizing augmentations based on configuration
        min_scale, max_scale = 2 ** (-self.scale_ratio), 2**self.scale_ratio
        augs.append(
            A.ResizeShortestEdge(
                short_edge_length=[int(x * self.resolution) for x in [min_scale, max_scale]],
                sample_style="range",
                max_size=self.max_size,
            )
        )

        ### Add rotation augmentations if configured
        if self.rotation > 0:
            angle = self.rotation * 180
            augs.append(A.RandomRotation(angle=(-angle, angle), expand=False))

        # Add cropping if configured
        if self.crop:
            crop_range = (self.crop_size or self.resolution, self.crop_size or self.resolution)
            augs.append(A.RandomCrop(crop_type="absolute_range", crop_size=crop_range))

        return augs


fai_instance_train_augs = DatasetAugmentations(
    resolution=1024,
    crop=True,
    scale_ratio=1.0,  # 0.5, 2
    max_size=2048,
    horizontal_flip=0.5,
    color_augmentation=1.0,
)

fai_segmentation_train_augs = DatasetAugmentations(
    resolution=640,
    crop=True,
    scale_ratio=1.0,  # 0.5, 2
    max_size=2048,
    color_augmentation=1.0,
    horizontal_flip=0.5,
)

fai_detection_train_augs = DatasetAugmentations(
    resolution=640,
    color_augmentation=1.0,
    horizontal_flip=0.5,
    aspect_ratio=0.5,  # 0.7, 1.4
    zoom_out=0.5,
    zoom_out_side=4.0,
    square=1.0,
    scale_ratio=0.5,  # 0.7, 1.4
)

detection_train_augs = DatasetAugmentations(
    resolution=640,
    square=1.0,
    max_size=int(640 * 1.25),
    crop=True,
    scale_ratio=0.5,  # 0.7, 1.4
    color_augmentation=1.0,
    horizontal_flip=0.5,
)

segmentation_train_augs = DatasetAugmentations(
    resolution=640,
    crop=True,
    scale_ratio=0.5,  # 0.7, 1.4
    color_augmentation=1.0,
    horizontal_flip=0.5,
)


detection_val_augs = DatasetAugmentations(
    resolution=640,
    square=1.0,
    max_size=int(640 * 1.25),
)

segmentation_val_augs = DatasetAugmentations(
    resolution=640,
    max_size=int(640 * 1.25),
)

classification_train_augs = DatasetAugmentations(
    resolution=224,
    scale_ratio=0.5,
    crop=True,
    color_augmentation=1.0,
    horizontal_flip=0.5,
)

classification_val_augs = DatasetAugmentations(
    resolution=224,
)


def get_default_by_task(
    task: Task, resolution: int = 640, advanced: bool = False
) -> Tuple[DatasetAugmentations, DatasetAugmentations]:
    if task == Task.DETECTION:
        train, val = (
            detection_train_augs if not advanced else fai_detection_train_augs,
            detection_val_augs if not advanced else detection_val_augs,
        )
    elif task == Task.SEMSEG:  # or task == Task.PANSEG:
        train, val = (
            segmentation_train_augs if not advanced else fai_segmentation_train_augs,
            segmentation_val_augs if not advanced else segmentation_val_augs,
        )
    elif task == Task.INSTANCE_SEGMENTATION:
        train, val = (
            segmentation_train_augs if not advanced else fai_instance_train_augs,
            segmentation_val_augs if not advanced else segmentation_val_augs,
        )
    elif task == Task.CLASSIFICATION:
        train, val = (
            classification_train_augs,
            classification_val_augs,
        )
    else:
        raise ValueError(f"Invalid task: {task}")

    train.resolution = resolution
    val.resolution = resolution
    return copy.deepcopy(train), copy.deepcopy(val)
