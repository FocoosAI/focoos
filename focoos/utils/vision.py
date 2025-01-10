import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
import supervision as sv
from scipy.ndimage import zoom
from typing_extensions import Buffer

from focoos.ports import FocoosDet, FocoosDetections


def index_to_class(class_ids: list[int], classes: list[str]) -> list[str]:
    return [classes[i] for i in class_ids]


def class_to_index(classes: list[str], class_names: list[str]) -> list[int]:
    # TODO: improve time complexity from O(mn) to O(n)
    return [class_names.index(c) for c in classes]


def image_loader(im: Union[bytes, str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads an image from various input types and converts it into a NumPy array
    suitable for processing with OpenCV.

    Args:
        im (Union[bytes, str, Path, np.ndarray, Image.Image]): The input image,
        which can be one of the following types:
            - bytes: Image data in raw byte format.
            - str: File path to the image.
            - Path: File path (Path object) to the image.
            - np.ndarray: Image in NumPy array format.
            - Image.Image: Image in PIL (Pillow) format.

    Returns:
        np.ndarray: The loaded image as a NumPy array, in BGR format, suitable for OpenCV processing.

    Raises:
        ValueError: If the input type is not one of the accepted types.
    """
    if isinstance(im, np.ndarray):
        cv_image = im
    elif isinstance(im, str) or isinstance(im, Path):
        cv_image = cv2.imread(str(im))
    elif isinstance(im, Image.Image):
        cv_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    elif isinstance(im, Buffer):
        byte_array = np.frombuffer(im, dtype=np.uint8)
        cv_image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

    return cv_image


def image_preprocess(
    im: Union[bytes, str, Path, np.ndarray, Image.Image],
    dtype=np.float32,
    resize: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses an image by loading it, optionally resizing it, and converting it into a
    format suitable for model input.

    Args:
        im (Union[bytes, str, Path, np.ndarray, Image.Image]): The input image. Can be a byte string,
            file path, file path as a string, NumPy array, or a PIL Image.
        dtype (np.dtype, optional): The desired data type of the output image tensor. Default is np.float32.
        resize (Optional[int], optional): If provided, the image will be resized to this dimension
            (height and width). Default is None, meaning no resizing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two elements:
            - The preprocessed image in the form of a NumPy array with shape (1, C, H, W) (CHW format).
            - The original loaded image in its raw form (height x width x channels).

    Example:
        im1, im0 = image_preprocess("image.jpg", resize=256)
    """
    im0 = image_loader(im)
    _im1 = im0
    if resize:
        _im1 = cv2.resize(im0, (resize, resize))

    im1 = np.ascontiguousarray(_im1.transpose(2, 0, 1)[np.newaxis, :]).astype(
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
    detections: sv.Detections, in_shape: tuple, out_shape: tuple
) -> sv.Detections:
    if in_shape[0] == out_shape[0] and in_shape[1] == out_shape[1]:
        return detections
    if detections.xyxy is not None:
        x_ratio = out_shape[0] / in_shape[0]
        y_ratio = out_shape[1] / in_shape[1]
        detections.xyxy = detections.xyxy * np.array(
            [x_ratio, y_ratio, x_ratio, y_ratio]
        )
    return detections


def base64mask_to_mask(base64mask: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(base64mask))))


def focoos_detections_to_supervision(
    inference_output: FocoosDetections,
) -> sv.Detections:
    xyxy = np.array(
        [
            d.bbox if d.bbox is not None else np.empty(4)
            for d in inference_output.detections
        ]
    )
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
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=masks,
    )


def binary_mask_to_base64(binary_mask: np.ndarray) -> str:
    """
    Converts a binary mask (NumPy array) to a base64-encoded PNG image.

    This function takes a binary mask, where values of `True` represent the areas of interest (usually 1s)
    and `False` represents the background (usually 0s). The binary mask is then converted to an image,
    and this image is saved in PNG format and encoded into a base64 string.

    Args:
        binary_mask (np.ndarray): A 2D NumPy array with boolean values (`True`/`False`).

    Returns:
        str: A base64-encoded string representing the PNG image of the binary mask.
    """
    # Convert the binary mask to uint8 type, then multiply by 255 to set True values to 255 (white)
    # and False values to 0 (black).
    binary_mask = binary_mask.astype(np.uint8) * 255

    # Create a PIL image from the NumPy array
    binary_mask_image = Image.fromarray(binary_mask)

    # Save the image to an in-memory buffer as PNG
    with io.BytesIO() as buffer:
        binary_mask_image.save(buffer, bitmap_format="png", format="PNG")
        # Get the PNG image in binary form and encode it to base64
        encoded_png = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded_png


def sv_to_focoos_detections(
    detections: sv.Detections, classes: Optional[list[str]] = None
) -> FocoosDetections:
    """
    Convert a list of detections from the supervision format to Focoos detection format.

    Args:
        detections (sv.Detections): A list of detections, where each detection is
            represented by a tuple containing bounding box coordinates (xyxy), mask,
            confidence score, class ID, track ID, and additional information.
        classes (Optional[list[str]], optional): A list of class labels. If provided,
            the function will map the class ID to the corresponding label. Defaults to None.

    Returns:
        FocoosDetections: An object containing a list of detections converted to the
            Focoos detection format.

    Notes:
        - The bounding box coordinates (xyxy) will be rounded to two decimal places.
        - If the mask is provided, it will be converted to a base64-encoded string.
        - Confidence score will be rounded to two decimal places.
        - If the class ID is valid and the class labels list is provided, the corresponding
          label will be assigned to each detection.

    Example:
        detections = sv.Detections([...])  # List of detections in supervision format
        classes = ['person', 'car', 'bike']
        result = sv_to_focoos_detections(detections, classes)
    """
    res = []
    for xyxy, mask, conf, cls_id, _, _ in detections:
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
