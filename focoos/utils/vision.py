import base64
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
import supervision as sv
from scipy.ndimage import zoom
from typing_extensions import Buffer

from focoos.ports import FocoosDet, FocoosDetections, FocoosTask


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

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


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

    im1 = np.ascontiguousarray(_im1.transpose(2, 0, 1)[np.newaxis, :]).astype(dtype)  # HWC->1CHW
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


def scale_detections(detections: sv.Detections, in_shape: tuple, out_shape: tuple) -> sv.Detections:
    if in_shape[0] == out_shape[0] and in_shape[1] == out_shape[1]:
        return detections
    if detections.xyxy is not None:
        x_ratio = out_shape[0] / in_shape[0]
        y_ratio = out_shape[1] / in_shape[1]
        detections.xyxy = detections.xyxy * np.array([x_ratio, y_ratio, x_ratio, y_ratio])

    if detections.mask is not None:
        detections.mask = np.array([scale_mask(m, out_shape) for m in detections.mask])
    return detections


def base64mask_to_mask(base64mask: str) -> np.ndarray:
    """
    Convert a base64-encoded mask to a binary mask using OpenCV.

    Args:
        base64mask (str): Base64-encoded string representing the mask.

    Returns:
        np.ndarray: Decoded binary mask as a NumPy array.
    """
    # Decode the base64 string to bytes and convert to a NumPy array in one step
    np_arr = np.frombuffer(base64.b64decode(base64mask), np.uint8)
    # Decode the NumPy array to an image using OpenCV and convert to a binary mask in one step
    binary_mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE) > 0
    return binary_mask.astype(bool)


def fai_detections_to_sv(inference_output: FocoosDetections, im0_shape: tuple) -> sv.Detections:
    xyxy = np.array([d.bbox if d.bbox is not None else np.empty(4) for d in inference_output.detections])
    class_id = np.array([d.cls_id for d in inference_output.detections])
    confidence = np.array([d.conf for d in inference_output.detections])
    if xyxy.shape[0] == 0:
        xyxy = np.zeros((0, 4))
    _masks = []
    if len(inference_output.detections) > 0 and inference_output.detections[0].mask:
        _masks = [np.zeros(im0_shape, dtype=bool) for _ in inference_output.detections]
        for i, det in enumerate(inference_output.detections):
            if det.mask:
                mask = base64mask_to_mask(det.mask)
                if det.bbox is not None and not np.array_equal(det.bbox, [0, 0, 0, 0]):
                    x1, y1, x2, y2 = map(int, det.bbox)
                    y2, x2 = min(y2, _masks[i].shape[0]), min(x2, _masks[i].shape[1])
                    _masks[i][y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
                else:
                    _masks[i] = mask
    masks = np.array(_masks).astype(bool) if len(_masks) > 0 else None
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=masks,
    )


def binary_mask_to_base64(binary_mask: np.ndarray) -> str:
    """
    Converts a binary mask (NumPy array) to a base64-encoded PNG image using OpenCV.

    This function takes a binary mask, where values of `True` represent the areas of interest (usually 1s)
    and `False` represents the background (usually 0s). The binary mask is then converted to an image,
    and this image is saved in PNG format and encoded into a base64 string.

    Args:
        binary_mask (np.ndarray): A 2D NumPy array with boolean values (`True`/`False`).

    Returns:
        str: A base64-encoded string representing the PNG image of the binary mask.
    """
    # Directly convert the binary mask to uint8 and multiply by 255 in one step
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Use OpenCV to encode the image as PNG
    success, encoded_image = cv2.imencode(".png", binary_mask)
    if not success:
        raise ValueError("Failed to encode image")

    # Encode the image to base64
    return base64.b64encode(encoded_image).decode("utf-8")


def sv_to_fai_detections(detections: sv.Detections, classes: Optional[list[str]] = None) -> List[FocoosDet]:
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
        if mask is not None:
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(x1 - 1, 0)
            y1 = max(y1 - 1, 0)
            x2 = min(x2 + 2, mask.shape[1])
            y2 = min(y2 + 2, mask.shape[0])
            cropped_mask = mask[y1:y2, x1:x2]
            mask = binary_mask_to_base64(cropped_mask)

        det = FocoosDet(
            cls_id=int(cls_id) if cls_id is not None else None,
            bbox=[int(x) for x in xyxy],
            mask=mask,
            conf=round(float(conf), 2) if conf is not None else None,
            label=(classes[cls_id] if classes is not None and cls_id is not None else None),
        )
        res.append(det)
    return res


def masks_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """
    Converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Parameters:
        masks (np.ndarray): A 3D `np.array` of shape `(N, W, H)`
            containing 2D bool masks

    Returns:
        np.ndarray: A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask
    """
    # Vectorized approach to find bounding boxes
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    # Use np.any to quickly find rows and columns with True values
    for i, mask in enumerate(masks):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


def get_postprocess_fn(task: FocoosTask):
    if task == FocoosTask.INSTANCE_SEGMENTATION:
        return instance_postprocess
    elif task == FocoosTask.SEMSEG:
        return semseg_postprocess
    else:
        return det_postprocess


def det_postprocess(out: List[np.ndarray], im0_shape: Tuple[int, int], conf_threshold: float) -> sv.Detections:
    """
    Postprocesses the output of an object detection model and filters detections
    based on a confidence threshold.

    Args:
        out (List[np.ndarray]): The output of the detection model.
        im0_shape (Tuple[int, int]): The original shape of the input image (height, width).
        conf_threshold (float): The confidence threshold for filtering detections.

    Returns:
        sv.Detections: A sv.Detections object containing the filtered bounding boxes, class ids, and confidences.
    """
    cls_ids, boxes, confs = out
    boxes[:, 0::2] *= im0_shape[1]
    boxes[:, 1::2] *= im0_shape[0]
    high_conf_indices = (confs > conf_threshold).nonzero()

    return sv.Detections(
        xyxy=boxes[high_conf_indices].astype(int),
        class_id=cls_ids[high_conf_indices].astype(int),
        confidence=confs[high_conf_indices].astype(float),
    )


def semseg_postprocess(out: List[np.ndarray], im0_shape: Tuple[int, int], conf_threshold: float) -> sv.Detections:
    """
    Postprocesses the output of a semantic segmentation model and filters based
    on a confidence threshold, removing empty masks.

    Args:
        out (List[np.ndarray]): The output of the semantic segmentation model.
        conf_threshold (float): The confidence threshold for filtering detections.

    Returns:
        sv.Detections: A sv.Detections object containing the non-empty masks, class ids, and confidences.
    """
    cls_ids, mask, confs = out[0][0], out[1][0], out[2][0]
    masks = np.equal(mask, np.arange(len(cls_ids))[:, None, None])
    high_conf_indices = confs > conf_threshold
    masks = masks[high_conf_indices]
    cls_ids = cls_ids[high_conf_indices]
    confs = confs[high_conf_indices]

    if len(masks.shape) != 3:
        return sv.Detections(
            mask=None,
            xyxy=np.zeros((0, 4)),
            class_id=None,
            confidence=None,
        )
    # Filter out empty masks
    non_empty_mask_indices = np.any(masks, axis=(1, 2))
    masks = masks[non_empty_mask_indices]
    cls_ids = cls_ids[non_empty_mask_indices]
    confs = confs[non_empty_mask_indices]
    xyxy = masks_to_xyxy(masks)
    return sv.Detections(
        mask=masks,
        # xyxy is required from supervision
        xyxy=xyxy,
        class_id=cls_ids,
        confidence=confs,
    )


def instance_postprocess(out: List[np.ndarray], im0_shape: Tuple[int, int], conf_threshold: float) -> sv.Detections:
    """
    Postprocesses the output of an instance segmentation model and filters detections
    based on a confidence threshold.
    """
    cls_ids, mask, confs = out[0][0], out[1][0], out[2][0]
    high_conf_indices = np.where(confs > conf_threshold)[0]
    masks = mask[high_conf_indices].astype(bool)
    cls_ids = cls_ids[high_conf_indices].astype(int)
    confs = confs[high_conf_indices].astype(float)
    if len(masks.shape) != 3:
        return sv.Detections(
            mask=None,
            xyxy=np.zeros((0, 4)),
            class_id=None,
            confidence=None,
        )

    # Filter out empty masks
    non_empty_mask_indices = np.any(masks, axis=(1, 2))
    masks = masks[non_empty_mask_indices]
    cls_ids = cls_ids[non_empty_mask_indices]
    confs = confs[non_empty_mask_indices]
    xyxy = masks_to_xyxy(masks)

    return sv.Detections(
        mask=masks,
        # xyxy is required from supervision
        xyxy=xyxy,
        class_id=cls_ids,
        confidence=confs,
    )
