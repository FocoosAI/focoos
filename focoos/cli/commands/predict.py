"""Predict command implementation.

This module implements the prediction/inference command for the Focoos CLI.
It provides functionality to run inference on images and videos using trained
computer vision models, with options to save annotations, JSON results, and masks.
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from focoos.model_manager import ModelManager
from focoos.ports import PREDICT_DIR, RuntimeType
from focoos.utils.logger import get_logger
from focoos.utils.vision import annotate_image, image_loader

logger = get_logger("predict")


def predict_command(
    model_name: str,
    source: str,
    runtime: RuntimeType = RuntimeType.ONNX_CUDA32,
    im_size: Optional[int] = 640,
    conf: Optional[float] = 0.5,
    save: Optional[bool] = True,
    output_dir: Optional[str] = PREDICT_DIR,
    save_json: Optional[bool] = True,
    save_masks: Optional[bool] = True,
    models_dir: Optional[str] = None,
):
    """Run inference on images or videos using a specified model.

    Loads a model, processes the input source (image/video), runs inference,
    and optionally saves results including annotations, JSON detections, and masks.
    Results are always printed to console.

    Args:
        model_name (str): Name of the model to use for inference.
        source (str): Path to image/video file or URL to process.
        runtime (RuntimeType, optional): Runtime type for inference.
            Defaults to RuntimeType.ONNX_CUDA32.
        im_size (Optional[int], optional): Input image size for the model.
            Defaults to 640.
        conf (Optional[float], optional): Confidence threshold for detections.
            Defaults to 0.5.
        save (Optional[bool], optional): Whether to save annotated images.
            Defaults to True.
        output_dir (Optional[str], optional): Directory to save results.
            Defaults to PREDICT_DIR.
        save_json (Optional[bool], optional): Whether to save detections as JSON.
            Defaults to True.
        save_masks (Optional[bool], optional): Whether to save masks as separate images.
            Defaults to True.

    Raises:
        Exception: If model loading, image loading, or inference fails.
    """
    logger.info(f"🔮 Starting prediction - Model: {model_name}, Source: {source}, Image size: {im_size}, Conf: {conf}")

    try:
        # Load image and run inference
        if source.startswith(("http://", "https://")):
            import requests

            response = requests.get(source)
            response.raise_for_status()
            image = image_loader(response.content)
        else:
            image = image_loader(source)
        model = ModelManager.get(model_name, models_dir=models_dir)
        exported_model = model.export(runtime_type=runtime, image_size=im_size)
        results = exported_model.infer(image, threshold=conf)

        # Print detections to console by default
        print_detections(results, model.classes)

        # Handle saving based on arguments
        if save or save_json or save_masks:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                source_path = Path(source)
                base_name = source_path.stem

                # Save annotated image
                if save:
                    annotated_image = annotate_image(
                        im=image, detections=results, task=model.task, classes=model.classes
                    )
                    save_path = os.path.join(output_dir, f"{base_name}_annotated{source_path.suffix}")
                    logger.info(f"Saving annotated image to {save_path}")
                    annotated_image.save(save_path)

                # Save detections as JSON
                if save_json:
                    json_path = os.path.join(output_dir, f"{base_name}_detections.json")
                    save_detections_json(results, json_path, model.classes)
                    logger.info(f"Saving detections to {json_path}")

                # Save masks if present and requested
                if save_masks and has_masks(results):
                    save_masks_as_images(results, output_dir, base_name)
                    logger.info(f"Saving masks to {output_dir}")

        logger.info("✅ Prediction completed!")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def print_detections(results, classes):
    """Print detection results to console in a formatted table.

    Args:
        results: Detection results from model inference.
        classes: List of class names for label mapping.
    """
    print("\n" + "=" * 50)
    print("DETECTION RESULTS")
    print("=" * 50)

    # Handle FocoosDetections structure
    if hasattr(results, "detections"):
        detections = results.detections
        num_detections = len(detections)
        print(f"Found {num_detections} detections:")
        print()

        for i, det in enumerate(detections):
            # Get values from FocoosDet object
            bbox = det.bbox if det.bbox else [0, 0, 0, 0]
            x1, y1, x2, y2 = bbox
            conf = det.conf if det.conf is not None else 0.0
            cls_id = det.cls_id if det.cls_id is not None else 0
            label = (
                det.label
                if det.label
                else (classes[cls_id] if classes and cls_id < len(classes) else f"class_{cls_id}")
            )

            print(f"  {i + 1}. {label}")
            print(f"     Confidence: {conf:.3f}")
            print(f"     Bbox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"     Size: {x2 - x1} x {y2 - y1}")
            if det.mask:
                print("     Has mask: Yes (base64 encoded)")
            print()
    else:
        print("No detections found.")

    print("=" * 50 + "\n")


def save_detections_json(results, json_path: str, classes):
    """Save detection results as a structured JSON file.

    Args:
        results: Detection results from model inference.
        json_path (str): Path where to save the JSON file.
        classes: List of class names for label mapping.
    """
    detections_data = {"detections": [], "summary": {"total_detections": 0, "classes_detected": []}}

    # Handle FocoosDetections structure
    if hasattr(results, "detections"):
        detections = results.detections
        num_detections = len(detections)
        detections_data["summary"]["total_detections"] = num_detections
        classes_found = set()

        for i, det in enumerate(detections):
            # Get values from FocoosDet object
            bbox = det.bbox if det.bbox else [0, 0, 0, 0]
            x1, y1, x2, y2 = bbox
            conf = float(det.conf) if det.conf is not None else 0.0
            cls_id = int(det.cls_id) if det.cls_id is not None else 0
            label = (
                det.label
                if det.label
                else (classes[cls_id] if classes and cls_id < len(classes) else f"class_{cls_id}")
            )
            classes_found.add(label)

            detection = {
                "id": i + 1,
                "class_id": cls_id,
                "class_name": label,
                "confidence": conf,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
            }

            # Add mask info if available (base64 encoded)
            if det.mask:
                detection["has_mask"] = True
                detection["mask_base64"] = det.mask

            detections_data["detections"].append(detection)

        detections_data["summary"]["classes_detected"] = list(classes_found)

    # Save to JSON file
    with open(json_path, "w") as f:
        json.dump(detections_data, f, indent=2)


def has_masks(results) -> bool:
    """Check if detection results contain mask information.

    Args:
        results: Detection results from model inference.

    Returns:
        bool: True if any detection contains mask data, False otherwise.
    """
    if hasattr(results, "detections"):
        return any(det.mask for det in results.detections)
    return False


def save_masks_as_images(results, save_dir: str, base_name: str):
    """Save detection masks as separate image files.

    Decodes base64 encoded masks from detection results and saves them as
    individual PNG image files in a dedicated masks directory.

    Args:
        results: Detection results containing base64 encoded masks.
        save_dir (str): Directory where to create the masks subdirectory.
        base_name (str): Base name for the mask files.
    """
    if not has_masks(results):
        return

    masks_dir = os.path.join(save_dir, f"{base_name}_masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Handle FocoosDetections structure
    if hasattr(results, "detections"):
        mask_count = 0
        for i, det in enumerate(results.detections):
            if det.mask:
                try:
                    mask_count += 1
                    # Decode base64 string to image
                    mask_string = det.mask

                    # Remove data URL prefix if present
                    if mask_string.startswith("data:image"):
                        mask_string = mask_string.split(",", 1)[1]

                    # Decode base64
                    mask_bytes = base64.b64decode(mask_string)
                    mask_image = Image.open(BytesIO(mask_bytes))

                    # Convert to grayscale if needed
                    if mask_image.mode != "L":
                        mask_image = mask_image.convert("L")

                    # Save as PNG
                    mask_path = os.path.join(masks_dir, f"mask_{mask_count}.png")
                    mask_image.save(mask_path)
                    logger.info(f"Saved mask {mask_count} (detection {i + 1}) to {mask_path}")

                except Exception as e:
                    logger.error(f"Failed to save mask for detection {i + 1}: {e}")
                    continue
