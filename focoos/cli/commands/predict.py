"""Prediction/Inference command implementation.

This module implements the prediction and inference command for the Focoos CLI.
It provides comprehensive functionality to run computer vision inference on images using trained models, with extensive options for result visualization,
annotation, and output format customization.

**Key Features:**
- **Flexible Output Formats**: Annotated images, JSON results, mask extraction
- **Real-time Visualization**: Console output with formatted detection results
- **Customizable Inference**: Configurable confidence thresholds and image sizes
- **Runtime Selection**: Support for different inference backends

**Supported Input Sources:**
- **Local Files**: Individual images
- **URLs**: Direct inference from web-hosted media

**Output Formats:**
- **Annotated Images**: Visual results with bounding boxes and labels
- **JSON Results**: Structured detection data with confidence scores
- **Mask Images**: Segmentation masks as separate image files
- **Console Output**: Formatted table display of detection results

**Use Cases:**
- Model evaluation and testing
- Result visualization and analysis
- Dataset annotation validation
- Production inference pipelines

Examples:
    Basic image inference:
    ```bash
    focoos predict --model fai-detr-m-coco --source image.jpg
    ```

    Programmatic usage:
    ```python
    from focoos.cli.commands import predict_command

    predict_command(
        model_name="fai-detr-m-coco",
        source="image.jpg",
        conf=0.5,
        save=True
    )
    ```

See Also:
    - [`focoos.model_manager.ModelManager`][focoos.model_manager.ModelManager]: Model loading and management
    - [`focoos.ports.FocoosDetections`][focoos.ports.FocoosDetections]: Detection result format
    - [`focoos.utils.vision.annotate_image`][focoos.utils.vision.annotate_image]: Image annotation utilities
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image

from focoos.model_manager import ModelManager
from focoos.ports import PREDICTIONS_DIR, FocoosDetections, RuntimeType
from focoos.utils.logger import get_logger
from focoos.utils.vision import image_loader

logger = get_logger("predict")


def predict_command(
    model_name: str,
    source: str,
    conf: float = 0.5,
    runtime: Optional[RuntimeType] = None,
    im_size: Optional[int] = 640,
    save: Optional[bool] = True,
    output_dir: Optional[str] = PREDICTIONS_DIR,
    save_json: Optional[bool] = True,
    save_masks: Optional[bool] = True,
):
    """Run inference on images or videos using a specified computer vision model.

    Loads a model, processes the input source (image, URL, or directory),
    runs inference with the specified parameters, and optionally saves results
    in multiple formats. Results are always displayed in the console with detailed
    detection information.

    **Inference Process:**
    1. **Input Loading**: Loads and preprocesses the input source
    2. **Model Loading**: Initializes the specified model with optional runtime
    3. **Inference Execution**: Runs prediction with confidence filtering
    4. **Result Processing**: Formats and displays detection results
    5. **Output Generation**: Saves results in requested formats

    **Output Formats:**
    - **Console Display**: Formatted table with detection details
    - **Annotated Images**: Visual results with bounding boxes and labels
    - **JSON Files**: Structured detection data with metadata
    - **Mask Images**: Segmentation masks as separate PNG files

    Args:
        model_name (str): Name or identifier of the model to use for inference.
            Can be a pretrained model name (e.g., "fai-detr-m-coco") or
            path to a local model checkpoint.
        source (str): Input source for inference. Supported formats:
            - Image files: .jpg, .jpeg, .png
            - URLs: Direct links to images
        runtime (Optional[RuntimeType], optional): Runtime backend for inference.
            If None, uses the model's default runtime. Options include:
            - RuntimeType.ONNX_CUDA32: ONNX with CUDA acceleration
            - RuntimeType.TORCHSCRIPT_32: TorchScript backend
            - RuntimeType.ONNX_CPU: CPU-only ONNX inference
            Defaults to None.
        im_size (Optional[int], optional): Input image size for model inference.
            Images are resized to this size while maintaining aspect ratio.
            Larger sizes may improve accuracy but increase processing time.
            Defaults to 640.
        conf (Optional[float], optional): Confidence threshold for detections.
            Only detections with confidence above this value are included.
            Range: 0.0 to 1.0. Lower values include more detections.
            Defaults to 0.5.
        save (Optional[bool], optional): Whether to save annotated images with
            bounding boxes, labels, and confidence scores overlaid.
            Defaults to True.
        output_dir (Optional[str], optional): Directory to save all output files.
            If None, uses the default predictions directory.
            Directory is created if it doesn't exist.
            Defaults to PREDICTIONS_DIR.
        save_json (Optional[bool], optional): Whether to save detection results
            as structured JSON files with complete metadata.
            Defaults to True.
        save_masks (Optional[bool], optional): Whether to save segmentation masks
            as separate PNG image files. Only applies to segmentation models.
            Defaults to True.

    Raises:
        Exception: If model loading, image loading, or inference fails.
        FileNotFoundError: If specified model or source file cannot be found.
        ValueError: If invalid parameters are provided (e.g., conf not in [0,1]).
        RuntimeError: If specified runtime is not available.

    Examples:
        Basic image inference:
        ```python
        predict_command(model_name="fai-detr-m-coco", source="image.jpg")
        ```

        High-confidence detections only:
        ```python
        predict_command(model_name="fai-detr-m-coco", source="image.jpg", conf=0.8, save_json=False)
        ```

    Note:
        - Console output is always generated regardless of save settings
        - Mask saving only applies to segmentation models
        - Large images may require significant processing time

    See Also:
        - [`focoos.model_manager.ModelManager.get`][focoos.model_manager.ModelManager.get]: Model loading
        - [`focoos.ports.FocoosDetections`][focoos.ports.FocoosDetections]: Detection result format
        - [`focoos.utils.vision.image_loader`][focoos.utils.vision.image_loader]: Image loading utilities
    """
    logger.info(f"🔄 Loading source: {source}")
    image = image_loader(source)

    logger.info(f"🔄 Loading model: {model_name}")
    model = ModelManager.get(model_name)
    if save or save_masks:
        annotate = True
    else:
        annotate = False
    if runtime:
        logger.info(f"🚀 Using runtime: {runtime}")
        exported_model = model.export(runtime_type=runtime, image_size=im_size)
        results = exported_model.infer(image, threshold=conf, annotate=annotate)
    else:
        logger.info("🚀 Using default model runtime")
        results = model.infer(image, threshold=conf, annotate=annotate)

    # Print detections to console by default
    print_detections(results)

    # Handle saving based on arguments
    if save or save_json or save_masks:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            source_path = Path(source)
            base_name = source_path.stem

            # Save annotated image
            if save and results.image is not None:
                annotated_image = results.image
                save_path = os.path.join(output_dir, f"{base_name}_annotated{source_path.suffix}")
                logger.info(f"💾 Saving annotated image to {save_path}")
                cv2.imwrite(save_path, annotated_image)  # type: ignore
                # annotated_image.save(save_path)

            # Save detections as JSON
            if save_json:
                json_path = os.path.join(output_dir, f"{base_name}_detections.json")
                save_detections_json(results, json_path)
                logger.info(f"💾 Saving detections to {json_path}")

            # Save masks if present and requested
            if save_masks and has_masks(results):
                save_masks_as_images(results, output_dir, base_name)
                logger.info(f"💾 Saving masks to {output_dir}")

    logger.info("✅ Prediction completed successfully!")


def print_detections(results: FocoosDetections):
    """Print detection results to console in a formatted table.

    Displays detection results in a user-friendly table format with detailed
    information about each detection including bounding boxes, confidence scores,
    class labels, and mask availability. Also shows inference latency metrics
    if available.

    Args:
        results (FocoosDetections): Detection results from model inference
            containing detection objects and optional timing information.

    Examples:
        Output format:
        ```
        ==================================================
        DETECTION RESULTS
        ==================================================
        Found 3 detections:

          1. person
             Confidence: 0.892
             Bbox: [120, 50, 300, 400]
             Size: 180 x 350
             Has mask: Yes (base64 encoded)

          2. car
             Confidence: 0.756
             Bbox: [450, 200, 650, 320]
             Size: 200 x 120

        Latencies:
        --------------------------------------------------
          preprocessing: 0.005s
          inference: 0.045s
          postprocessing: 0.012s
        ==================================================
        ```
    """
    print("\n" + "=" * 50)
    print("DETECTION RESULTS")
    print("=" * 50)

    detections = results.detections
    num_detections = len(detections)
    print(f"Found {num_detections} detections{': ' if num_detections > 0 else ''}")
    print()

    for i, det in enumerate(detections):
        # Get values from FocoosDet object
        x1, y1, x2, y2 = det.bbox if det.bbox else [-1, -1, -1, -1]
        conf = det.conf if det.conf is not None else -1

        print(f"  {i + 1}. {det.label or f'Class {det.cls_id}'}")
        print(f"     Confidence: {conf:.3f}")
        print(f"     Bbox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"     Size: {x2 - x1} x {y2 - y1}")
        if det.mask:
            print("     Has mask: Yes (base64 encoded)")
        print()

    # Print latency information if available
    if results.latency:
        print("Latencies:")
        print("-" * 50)
        for key, value in results.latency.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}s")
            else:
                print(f"  {key}: {value}")
    print()
    print("=" * 50 + "\n")


def save_detections_json(results: FocoosDetections, json_path: str):
    """Save detection results as a structured JSON file.

    Exports detection results to a JSON file with comprehensive metadata
    including detection details, summary statistics, and timing information.
    The JSON format is suitable for further processing, analysis, or
    integration with other systems.

    Args:
        results (FocoosDetections): Detection results from model inference
            containing all detection data and metadata.
        json_path (str): Path where to save the JSON file.
            Parent directories are created if they don't exist.

    JSON Structure:
        ```json
        {
          "detections": [
            {
              "bbox": [x1, y1, x2, y2],
              "conf": 0.85,
              "cls_id": 0,
              "label": "person",
              "mask": "base64_string_if_available"
            }
          ],
          "summary": {
            "total_detections": 2,
            "classes_detected": 1
          },
          "latency": {
            "preprocessing": 0.005,
            "inference": 0.045,
            "postprocessing": 0.012
          }
        }
        ```

    Raises:
        IOError: If the file cannot be written to the specified path.
        PermissionError: If write permissions are insufficient.
    """
    detections_data = {"detections": [], "summary": {"total_detections": 0, "classes_detected": 0}}

    detections = results.detections
    num_detections = len(detections)
    detections_data["summary"]["total_detections"] = num_detections

    # Convert detections to dictionary format
    results_dict = results.model_dump()
    detections_data["detections"] = results_dict["detections"]

    # Count unique classes
    unique_classes = set(det.cls_id for det in detections if det.cls_id is not None)
    detections_data["summary"]["classes_detected"] = len(unique_classes)

    # Add latency information if available
    if results.latency:
        detections_data["latency"] = results.latency

    # Ensure output directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Save to JSON file
    with open(json_path, "w") as f:
        json.dump(detections_data, f, indent=2)


def has_masks(results: FocoosDetections) -> bool:
    """Check if detection results contain segmentation mask information.

    Determines whether any of the detections in the results include
    segmentation masks, which is useful for conditional mask processing
    and output generation.

    Args:
        results (FocoosDetections): Detection results from model inference
            to check for mask presence.

    Returns:
        bool: True if any detection contains mask data, False otherwise.

    Examples:
        ```python
        if has_masks(results):
            print("Segmentation masks available")
            save_masks_as_images(results, output_dir, base_name)
        else:
            print("No segmentation masks found")
        ```
    """
    return any(det.mask for det in results.detections)


def save_masks_as_images(results: FocoosDetections, save_dir: str, base_name: str):
    """Save detection masks as separate image files.

    Decodes base64 encoded segmentation masks from detection results and saves
    them as individual PNG image files in a dedicated masks directory. Each
    mask is saved with a sequential filename and includes error handling for
    corrupted or invalid mask data.

    **Mask Processing:**
    - Decodes base64 encoded mask strings
    - Handles data URL prefixes (data:image/png;base64,...)
    - Converts to grayscale format for consistency
    - Saves as PNG files with sequential naming

    Args:
        results (FocoosDetections): Detection results containing base64 encoded
            segmentation masks.
        save_dir (str): Directory where to create the masks subdirectory.
            The function creates a subdirectory named "{base_name}_masks".
        base_name (str): Base name for the mask files. Masks are saved as
            "mask_1.png", "mask_2.png", etc.

    Directory Structure:
        ```
        save_dir/
        └── {base_name}_masks/
            ├── mask_1.png
            ├── mask_2.png
            └── mask_3.png
        ```

    Raises:
        OSError: If the masks directory cannot be created.
        ValueError: If base64 decoding fails for any mask.

    Examples:
        ```python
        # Save masks for an image named "test.jpg"
        save_masks_as_images(results, "./output", "test")
        # Creates: ./output/test_masks/mask_1.png, etc.
        ```

    Note:
        - Only processes detections that have non-empty mask data
        - Handles both raw base64 strings and data URLs
        - Continues processing even if individual masks fail
        - Logs success and failure for each mask operation
    """
    if not has_masks(results):
        logger.warning("No masks found in detection results")
        return

    masks_dir = os.path.join(save_dir, f"{base_name}_masks")
    os.makedirs(masks_dir, exist_ok=True)

    mask_count = 0
    for i, det in enumerate(results.detections):
        if not det.mask:
            continue

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
            logger.info(f"💾 Saved mask {mask_count} (detection {i + 1}) to {mask_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save mask for detection {i + 1}: {e}")
            continue

    logger.info(f"✅ Saved {mask_count} masks to {masks_dir}")
