"""
Type definitions for Lightning DataModule.
Dataclasses per gli output tipizzati del dataset.
"""

from dataclasses import dataclass

import torch


@dataclass
class DetectionSample:
    """Output sample per detection/instance segmentation"""

    image: torch.Tensor  # (C, H, W)
    bboxes: torch.Tensor  # (N, 4) in formato [x_min, y_min, x_max, y_max]
    labels: torch.Tensor  # (N,)
    image_id: int
    file_name: str


@dataclass
class ClassificationSample:
    """Output sample per classification"""

    image: torch.Tensor  # (C, H, W)
    label: int
    image_id: int
    file_name: str


@dataclass
class GenericSample:
    """Output sample generico"""

    image: torch.Tensor  # (C, H, W)
    image_id: int
    file_name: str


@dataclass
class SegmentationSample:
    """Output sample per semantic segmentation"""

    image: torch.Tensor  # (C, H, W)
    mask: torch.Tensor  # (H, W) - maschera di segmentazione
    image_id: int
    file_name: str
    sem_seg_file_name: str
