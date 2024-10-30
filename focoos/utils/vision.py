from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
from supervision import Detections
from typing_extensions import Buffer

from focoos.ports import ModelMetadata


def index_to_class(class_ids: list[int], classes: list[str]) -> list[str]:
    return [classes[i] for i in class_ids]


def class_to_index(classes: list[str], class_names: list[str]) -> list[int]:
    return [class_names.index(c) for c in classes]


def focoos_det_to_supervision(detections: Detections) -> Detections:
    return Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
    )


def image_loader(im: Union[bytes, str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(im, Buffer):
        # Leggi il buffer in un array numpy
        byte_array = np.frombuffer(im, dtype=np.uint8)
        # Decodifica l'immagine usando OpenCV
        cv_image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

    elif isinstance(im, str) or isinstance(im, Path):
        cv_image = cv2.imread(str(im))

    elif isinstance(im, Image.Image):
        cv_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    elif isinstance(im, np.ndarray):
        cv_image = im

    return cv_image


def image_preprocess(
    im: Union[bytes, str, Path, np.ndarray, Image.Image],
    dtype=np.float32,
    resize: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    im0 = image_loader(im)
    if resize:
        res_image = cv2.resize(im0, (resize, resize))
        im1 = np.ascontiguousarray(
            res_image.transpose(2, 0, 1)[np.newaxis, :]  # HWC->1CHW
        ).astype(dtype)
        return im1, im0
    else:
        im1 = np.ascontiguousarray(
            im0.transpose(2, 0, 1)[np.newaxis, :]  # HWC->1CHW
        ).astype(dtype)
        return im1, im0
