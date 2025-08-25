import base64
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image
import supervision as sv
from scipy.ndimage import zoom
from torchvision.io import ImageReadMode
from torchvision.io.image import read_image
from typing_extensions import Buffer

from focoos.ports import CACHE_DIR, FocoosDet, FocoosDetections, Task
from focoos.utils.api_client import ApiClient

api_client = ApiClient()
focoos_color_palette = sv.ColorPalette.from_hex(["#015fe6", "#3faebd", "#63dba6", "#a151ff", "#df923a"])
label_annotator = sv.LabelAnnotator(color=focoos_color_palette, text_padding=10, border_radius=10, smart_position=False)
box_annotator = sv.BoxAnnotator(color=focoos_color_palette)
mask_annotator = sv.MaskAnnotator(color=focoos_color_palette)
edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN)
vertex_annotator = sv.VertexAnnotator(color=sv.Color.YELLOW)


def index_to_class(class_ids: list[int], classes: list[str]) -> list[str]:
    return [classes[i] for i in class_ids]


def class_to_index(classes: list[str], class_names: list[str]) -> list[int]:
    # TODO: improve time complexity from O(mn) to O(n)
    return [class_names.index(c) for c in classes]


def image_loader(im: Union[bytes, str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads an image from various input types and converts it into a NumPy array
    in RGB format suitable for processing.

    Args:
        im (Union[bytes, str, Path, np.ndarray, Image.Image]): The input image,
        which can be one of the following types:
            - bytes: Image data in raw byte format.
            - str: File path to the image.
            - Path: File path (Path object) to the image.
            - np.ndarray: Image in NumPy array format (assumed to be in RGB).
            - Image.Image: Image in PIL (Pillow) format.
            - URL: URL of the image (e.g. http:// or https://).

    Returns:
        np.ndarray: The loaded image as a NumPy array in RGB format.

    Raises:
        ValueError: If the input type is not one of the accepted types.
    """
    if isinstance(im, str) and im.startswith(("http://", "https://")):
        file_path = api_client.download_ext_file(im, CACHE_DIR, skip_if_exists=True)
        im = file_path

    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im, str) or isinstance(im, Path):
        return read_image(str(im), mode=ImageReadMode.RGB).permute(1, 2, 0).numpy()
    elif isinstance(im, Image.Image):
        cv_image = np.array(im)
    elif isinstance(im, Buffer):
        byte_array = np.frombuffer(im, dtype=np.uint8)
        cv_image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError("Could not decode image from buffer")

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


#!TODO DEPRECATED
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
            - The preprocessed image in the form of a NumPy array with shape (1, C, H, W) (CHW format) in RGB.
            - The original loaded image in RGB format (height x width x channels).

    Example:
        im1, im0 = image_preprocess("image.jpg", resize=256)
    """
    # Load image in RGB format
    im0 = image_loader(im)

    # Optimize: avoid unnecessary copy when no resize needed
    if resize is not None and (im0.shape[0] != resize or im0.shape[1] != resize):
        processed_im = cv2.resize(im0, (resize, resize), interpolation=cv2.INTER_LINEAR)
    else:
        processed_im = im0

    # Optimize: combine transpose, expand dims, and astype in one operation
    # HWC -> CHW -> 1CHW with efficient memory layout
    im1 = np.ascontiguousarray(processed_im.transpose(2, 0, 1)[np.newaxis, :], dtype=dtype)

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
    Optimized for better performance.

    Args:
        base64mask (str): Base64-encoded string representing the mask.

    Returns:
        np.ndarray: Decoded binary mask as a NumPy array.
    """
    # Optimization: decode and conversion in a single step
    try:
        # Decode base64 directly to bytes buffer
        img_bytes = base64.b64decode(base64mask)
        # Decode directly as uint8 array
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        # Decode and binarize in a single operation
        binary_mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        return binary_mask > 0  # Convert directly to bool
    except Exception:
        # Fallback in case of error
        np_arr = np.frombuffer(base64.b64decode(base64mask), np.uint8)
        binary_mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE) > 0
        return binary_mask.astype(bool)


def fai_detections_to_sv(inference_output: FocoosDetections, im0_shape: tuple) -> sv.Detections:
    # Early return if no detections
    if not inference_output.detections:
        return sv.Detections(
            xyxy=np.zeros((0, 4)),
            class_id=np.array([]),
            confidence=np.array([]),
            mask=None,
        )

    # Optimize array extraction using more efficient list comprehensions
    detections = inference_output.detections
    xyxy = np.array([d.bbox for d in detections if d.bbox is not None], dtype=np.float32)
    class_id = np.array([d.cls_id for d in detections], dtype=np.int32)
    confidence = np.array([d.conf for d in detections], dtype=np.float32)

    # If no valid bboxes, create empty array
    if xyxy.shape[0] == 0:
        xyxy = np.zeros((0, 4), dtype=np.float32)

    masks = None
    # Optimize mask handling - check only if necessary
    if detections and detections[0].mask:
        # Pre-allocate only if needed and use more efficient dtype
        _masks = []
        for i, det in enumerate(detections):
            if det.mask:
                mask = base64mask_to_mask(det.mask)
                full_mask = np.zeros(im0_shape, dtype=bool)

                if det.bbox is not None and not np.array_equal(det.bbox, [0, 0, 0, 0]):
                    x1, y1, x2, y2 = map(int, det.bbox)
                    # Optimized bound checking
                    y2, x2 = min(y2, im0_shape[0]), min(x2, im0_shape[1])
                    y1, x1 = max(0, y1), max(0, x1)

                    mask_h, mask_w = y2 - y1, x2 - x1
                    if mask_h > 0 and mask_w > 0:
                        full_mask[y1:y2, x1:x2] = mask[:mask_h, :mask_w]
                else:
                    # Ensure mask has correct dimensions
                    if mask.shape == im0_shape:
                        full_mask = mask

                _masks.append(full_mask)
            else:
                _masks.append(np.zeros(im0_shape, dtype=bool))

        masks = np.array(_masks, dtype=bool) if _masks else None

    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=masks,
    )


def fai_keypoints_to_sv(
    inference_output: FocoosDetections, im0_shape: tuple, keypoints_threshold: float = 0.5
) -> sv.KeyPoints:
    detections = inference_output.detections
    keypoints_data = np.array([d.keypoints for d in detections if d.keypoints is not None], dtype=np.float32)

    # Filter out keypoints with confidence below threshold
    confidence_values = keypoints_data[:, :, 2].copy()
    filter_mask = confidence_values < keypoints_threshold
    keypoints_data[filter_mask] = [0, 0, 0]

    # Extract xy coordinates (first two columns) and confidence (third column)
    keypoints_xy = keypoints_data[:, :, :2]  # Shape: (num_detections, num_keypoints, 2)
    keypoints_confidence = keypoints_data[:, :, 2]  # Shape: (num_detections, num_keypoints)

    class_id = np.array([d.cls_id for d in detections], dtype=int)

    return sv.KeyPoints(
        xy=keypoints_xy,
        class_id=class_id,
        confidence=keypoints_confidence,
    )


def trim_mask(mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    y2, x2 = min(y2, mask.shape[0]), min(x2, mask.shape[1])  # type: ignore
    return mask[y1:y2, x1:x2]


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
    return base64.b64encode(encoded_image.tobytes()).decode("utf-8")


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


#!TODO DEPRECATED
def annotate_image(
    im: Union[np.ndarray, Image.Image], detections: FocoosDetections, task: Task, classes: Optional[list[str]] = None
) -> Image.Image:
    if isinstance(im, Image.Image):
        im = np.array(im)

    sv_detections = fai_detections_to_sv(detections, im.shape[:2])
    # Optimized early return - check if there are detections
    if sv_detections.xyxy.shape[0] == 0:
        return Image.fromarray(im)

    # Use image directly without unnecessary copies
    annotated_im = im

    if task == Task.DETECTION:
        annotated_im = box_annotator.annotate(scene=annotated_im, detections=sv_detections)

    elif task in [
        Task.SEMSEG,
        Task.INSTANCE_SEGMENTATION,
    ]:
        annotated_im = mask_annotator.annotate(scene=annotated_im, detections=sv_detections)

    # Optimize label creation
    if classes is not None and sv_detections.class_id is not None and sv_detections.confidence is not None:
        labels = [
            f"{classes[int(class_id)]}: {confid * 100:.0f}%"
            for class_id, confid in zip(sv_detections.class_id, sv_detections.confidence)
        ]
        annotated_im = label_annotator.annotate(scene=annotated_im, detections=sv_detections, labels=labels)

    return Image.fromarray(annotated_im)


def annotate_frame(
    im: np.ndarray,
    detections: FocoosDetections,
    task: Task,
    classes: Optional[list[str]] = None,
    keypoints_skeleton: Optional[list[tuple[int, int]]] = None,
    keypoints_threshold: float = 0.5,
) -> np.ndarray:
    if isinstance(im, Image.Image):
        im = np.array(im)
    if len(detections.detections) == 0:
        return im
    has_bbox = detections.detections[0].bbox is not None
    has_mask = detections.detections[0].mask is not None
    has_keypoints = detections.detections[0].keypoints is not None
    if has_keypoints:
        sv_keypoints = fai_keypoints_to_sv(detections, im.shape[:2], keypoints_threshold=keypoints_threshold)
    sv_detections = fai_detections_to_sv(detections, im.shape[:2])
    if sv_detections.xyxy.shape[0] == 0:
        return im  # Return original RGB image

    annotated_im = im

    if has_bbox and task != Task.SEMSEG:
        annotated_im = box_annotator.annotate(scene=annotated_im, detections=sv_detections)

    if has_mask:
        annotated_im = mask_annotator.annotate(scene=annotated_im, detections=sv_detections)
    if has_keypoints:
        edge_annotator.edges = keypoints_skeleton

        annotated_im = edge_annotator.annotate(scene=annotated_im, key_points=sv_keypoints)
        annotated_im = vertex_annotator.annotate(scene=annotated_im, key_points=sv_keypoints)
    # Optimize label creation
    if classes is not None and sv_detections.class_id is not None and sv_detections.confidence is not None:
        labels = [
            f"{classes[int(class_id)]}: {confid * 100:.0f}%"
            for class_id, confid in zip(sv_detections.class_id, sv_detections.confidence)
        ]
        annotated_im = label_annotator.annotate(scene=annotated_im, detections=sv_detections, labels=labels)
    return annotated_im
