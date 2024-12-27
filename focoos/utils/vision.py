import base64
import io
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
from scipy.ndimage import zoom
from supervision import Detections
from typing_extensions import Buffer

from focoos.ports import FocoosDet, FocoosDetections, ModelMetadata


def index_to_class(class_ids: list[int], classes: list[str]) -> list[str]:
    return [classes[i] for i in class_ids]


def class_to_index(classes: list[str], class_names: list[str]) -> list[int]:
    # TODO: improve time complexity from O(mn) to O(n)
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
        im0 = cv2.resize(im0, (resize, resize))
    im1 = np.ascontiguousarray(im0.transpose(2, 0, 1)[np.newaxis, :]).astype(
        dtype
    )  # HWC->1CHW
    return im1, im0


def scale_mask(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Scales a binary mask to a target shape using nearest-neighbor interpolation.

    Args:
        mask (np.ndarray): Input binary mask array to be scaled.
        target_shape (tuple): Desired output shape as (height, width).

    Returns:
        np.ndarray: Scaled binary mask with the target shape.

    Example:
        >>> input_mask = np.array([[1, 0], [0, 1]], dtype=bool)
        >>> scaled = scale_mask(input_mask, (4, 4))
        >>> scaled.shape
        (4, 4)
    """
    # Calculate scale factors for height and width
    scale_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])

    # Resize the mask using zoom with nearest-neighbor interpolation (order=0)
    scaled_mask = zoom(mask, scale_factors, order=0) > 0.5

    return scaled_mask.astype(bool)


def scale_detections(
    detections: Detections, in_shape: tuple, out_shape: tuple
) -> Detections:
    if in_shape[0] == out_shape[0] and in_shape[1] == out_shape[1]:
        return detections
    t0 = time.time()
    if detections.xyxy is not None:
        x_ratio = out_shape[0] / in_shape[0]
        y_ratio = out_shape[1] / in_shape[1]
        detections.xyxy = detections.xyxy * np.array(
            [x_ratio, y_ratio, x_ratio, y_ratio]
        )
    t1 = time.time()
    return detections


def base64mask_to_mask(base64mask: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(base64mask))))


def focoos_detections_to_supervision(inference_output: FocoosDetections) -> Detections:
    xyxy = np.array([d.bbox for d in inference_output.detections])
    class_id = np.array([d.cls_id for d in inference_output.detections])
    confidence = np.array([d.conf for d in inference_output.detections])
    if xyxy.shape[0] == 0:
        xyxy = np.empty((0, 4))
    _masks = []
    for det in inference_output.detections:
        if det.mask:
            mask = base64mask_to_mask(det.mask)
            _masks.append(mask)
    masks = np.array(_masks).astype(bool) if len(_masks) > 0 else None
    return Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=masks,
    )


def binary_mask_to_base64(binary_mask):
    # Converti l'array NumPy in un oggetto immagine PIL
    # Converte True in 255 (bianco) e False in 0 (nero)
    binary_mask = binary_mask.astype(np.uint8) * 255
    binary_mask_image = Image.fromarray(binary_mask)

    # Salva l'immagine in memoria
    with io.BytesIO() as buffer:
        binary_mask_image.save(buffer, bitmap_format="png", format="PNG")
        encoded_png = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded_png


def sv_to_focoos_detections(
    detections: Detections, classes: Optional[list[str]] = None
) -> FocoosDetections:
    res = []
    for xyxy, mask, conf, cls_id, track_id, _ in detections:
        det = FocoosDet(
            cls_id=int(cls_id) if cls_id is not None else None,
            bbox=[round(float(x), 2) for x in xyxy],
            mask=binary_mask_to_base64(mask) if mask is not None else None,
            conf=round(float(conf), 2) if conf is not None else None,
            label=(
                classes[cls_id] if classes is not None and cls_id is not None else None
            ),
        )
        res.append(det)
    return FocoosDetections(detections=res)
