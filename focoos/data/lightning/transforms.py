"""
Custom Albumentations transforms for Lightning DataModule.
Include LetterBox (YOLO-style resize) e MosaicTransform.
"""

from typing import List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from numpy.typing import NDArray


class LetterBox(A.DualTransform):
    """
    LetterBox resize simile a Ultralytics/YOLO.
    Ridimensiona l'immagine mantenendo l'aspect ratio e aggiunge padding grigio.
    """

    def __init__(self, height: int = 640, width: int = 640, border_value: int = 114, p: float = 1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.border_value = border_value
        # Cache per dimensioni originali (usato da apply_to_bboxes)
        self._orig_h = None
        self._orig_w = None
        self._scale_ratio = None
        self._pad_top = None
        self._pad_left = None

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applica letterbox all'immagine"""
        h, w = img.shape[:2]

        # Salva dimensioni originali per apply_to_bboxes
        self._orig_h = h
        self._orig_w = w

        # Calcola il ratio di scaling
        r = min(self.height / h, self.width / w)
        new_h, new_w = int(h * r), int(w * r)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calcola padding
        dh, dw = self.height - new_h, self.width - new_w
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left

        # Salva parametri per apply_to_bboxes
        self._scale_ratio = r
        self._pad_top = top
        self._pad_left = left

        # Applica padding
        result = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.border_value,) * 3
        )

        return result

    def apply_to_bboxes(self, bboxes: Union[List[List[float]], NDArray], **params) -> NDArray[np.float32]:
        """Aggiusta le bounding boxes per letterbox"""
        if len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # Usa parametri salvati da apply()
        if self._orig_h is None or self._orig_w is None:
            # Fallback se apply non è stato chiamato - ritorna senza trasformazione
            if isinstance(bboxes, np.ndarray):
                return bboxes
            return np.array(bboxes, dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)

        h = self._orig_h
        w = self._orig_w
        r = self._scale_ratio
        left = self._pad_left
        top = self._pad_top

        adjusted_bboxes = []
        for bbox in bboxes:
            # Gestisce sia formato [x, y, w, h] che [x, y, w, h, class] o altri
            if len(bbox) >= 4:
                x_min, y_min, x_max, y_max = bbox[:4]
                extra = bbox[4:] if len(bbox) > 4 else []
            else:
                continue  # Skip bbox malformati

            # Denormalizza dalle coordinate originali
            x_min, x_max = x_min * w, x_max * w
            y_min, y_max = y_min * h, y_max * h
            # Scale e shift
            x_min = (x_min * r + left) / self.width
            x_max = (x_max * r + left) / self.width
            y_min = (y_min * r + top) / self.height
            y_max = (y_max * r + top) / self.height

            # Ricostruisci bbox con eventuali extra fields
            adjusted_bbox = [x_min, y_min, x_max, y_max] + list(extra)
            adjusted_bboxes.append(adjusted_bbox)

        # Ritorna numpy array invece di lista
        if len(adjusted_bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(adjusted_bboxes, dtype=np.float32)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("height", "width", "border_value")


class ToTensor(A.BasicTransform):
    """
    Convert image and mask to torch.Tensor WITHOUT dividing by 255.

    This keeps images in [0, 255] range, which is the expected format for processors
    that handle normalization with pixel_mean and pixel_std.

    Unlike albumentations.pytorch.ToTensorV2 which divides by 255, this transform
    maintains the original scale, allowing color augmentations to work naturally
    and deferring normalization to the processor.
    """

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: np.ndarray, **params) -> torch.Tensor:
        """
        Convert numpy image to torch tensor [C, H, W] in float32.
        Keeps values in [0, 255] range.
        """
        # HWC to CHW
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

    def apply_to_mask(self, mask: np.ndarray, **params) -> torch.Tensor:
        """Convert mask to torch tensor, keeping as long for segmentation."""
        return torch.from_numpy(mask).long()

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()

    def get_params_dependent_on_targets(self, params):
        return {}
