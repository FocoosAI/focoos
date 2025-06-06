"""Export command implementation.

This module implements the model export command for the Focoos CLI. It provides
functionality to export trained computer vision models to different formats
such as ONNX and TorchScript for deployment and production use.
"""

from typing import Literal, Optional

from focoos.model_manager import ModelManager
from focoos.ports import ExportFormat, RuntimeType
from focoos.utils.logger import get_logger

logger = get_logger("export")


def export_command(
    model_name: str,
    format: Optional[ExportFormat] = ExportFormat.TORCHSCRIPT,
    output_dir: Optional[str] = None,
    device: Optional[Literal["cuda", "cpu"]] = None,
    onnx_opset: Optional[int] = None,
    im_size: Optional[int] = None,
    overwrite: Optional[bool] = None,
    models_dir: Optional[str] = None,
):
    """Export a model to different formats.

    Loads a model and exports it to the specified format (ONNX or TorchScript).
    The exported model is saved to the specified output directory with appropriate
    runtime configuration based on the target format.

    Args:
        model_name (str): Name of the model to export.
        format (Optional[ExportFormat], optional): Export format to use.
            Defaults to ExportFormat.TORCHSCRIPT.
        output_dir (Optional[str], optional): Directory to save the exported model.
            Defaults to None.
        device (Optional[Literal["cuda", "cpu"]], optional): Device to use for export.
            Defaults to "cuda" if None.
        onnx_opset (Optional[int], optional): ONNX opset version to use for ONNX export.
            Defaults to 17 if None.
        im_size (Optional[int], optional): Input image size for the exported model.
            Defaults to None.
        overwrite (Optional[bool], optional): Whether to overwrite existing exported files.
            Defaults to False if None.

    Raises:
        Exception: If model loading fails or export process encounters an error.
    """
    logger.info(
        f"📦 Starting export - Model: {model_name}, Format: {format}, Output dir: {output_dir}, Device: {device}, ONNX opset: {onnx_opset}, Image size: {im_size}, Overwrite: {overwrite}"
    )

    try:
        # Initialize model
        logger.info(f"Loading model: {model_name}")
        model = ModelManager.get(name=model_name, models_dir=models_dir)

        # mapping default runtime
        runtime_type = RuntimeType.ONNX_CUDA32 if format == ExportFormat.ONNX else RuntimeType.TORCHSCRIPT_32

        # Export model
        logger.info(f"Exporting to {format} format...")
        export_path = model.export(
            runtime_type=runtime_type,
            onnx_opset=onnx_opset or 17,
            out_dir=output_dir,
            device=device or "cuda",
            overwrite=overwrite or False,
            image_size=im_size,
        )

        logger.info(f"✅ Model exported successfully to: {export_path}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
