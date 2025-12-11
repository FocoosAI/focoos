"""Export command implementation.

This module implements the model export command for the Focoos CLI. It provides
comprehensive functionality to export trained computer vision models to different
deployment formats such as ONNX and TorchScript for production use, edge deployment,
and cross-platform compatibility.

**Supported Export Formats:**
- **ONNX**: Open Neural Network Exchange format for cross-platform deployment
- **TorchScript**: PyTorch's production-ready serialization format

**Key Features:**
- **Cross-Platform Deployment**: Export models for use across different frameworks
- **Production Optimization**: Optimized models for inference performance
- **Edge Deployment**: Support for mobile and embedded device deployment
- **Flexible Configuration**: Customizable export parameters and optimization levels
- **Runtime Selection**: Automatic runtime type mapping based on export format

**Use Cases:**
- Production model deployment
- Cross-platform model sharing
- Edge device deployment
- Mobile application integration
- Web inference deployment
- Model optimization for serving

**Export Process:**
1. Model loading and validation
2. Runtime type selection based on format
3. Model compilation and optimization
4. Format-specific serialization
5. Export validation and saving

Examples:
    Basic ONNX export:
    ```bash
    focoos export --model fai-detr-m-coco
    ```

    TorchScript export with custom settings:
    ```bash
    focoos export --model fai-detr-m-coco --format torchscript --device cuda --im-size 1024
    ```

    Programmatic usage:
    ```python
    from focoos.cli.commands import export_command

    export_command(model_name="fai-detr-m-coco", format=ExportFormat.ONNX, device="cuda")
    ```

See Also:
    - [`focoos.model_manager.ModelManager`][focoos.model_manager.ModelManager]: Model loading and management
    - [`focoos.ports.ExportFormat`][focoos.ports.ExportFormat]: Available export formats
    - [`focoos.ports.RuntimeType`][focoos.ports.RuntimeType]: Runtime type configurations
"""

from typing import Literal, Optional, Tuple, Union

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
    im_size: Optional[Union[int, Tuple[int, int]]] = None,
    overwrite: Optional[bool] = None,
):
    """Export a model to different deployment formats.

    Loads a specified model and exports it to the chosen format (ONNX or TorchScript)
    with optimizations for production deployment. The exported model includes
    appropriate runtime configuration and is saved with proper file extensions
    and metadata.

    **Export Formats:**
    - **ONNX**: Cross-platform format supporting multiple inference engines
      - Supports ONNX Runtime, TensorRT, OpenVINO, and others
      - Optimized for production serving and edge deployment
      - Includes model graph optimization and quantization support

    - **TorchScript**: PyTorch native serialization format
      - JIT-compiled for optimized PyTorch inference
      - Maintains full PyTorch compatibility
      - Supports both scripting and tracing compilation

    **Runtime Mapping:**
    - ONNX format → ONNX_CUDA32 runtime (GPU) or ONNX_CPU (CPU)
    - TorchScript format → TORCHSCRIPT_32 runtime

    Args:
        model_name (str): Name or identifier of the model to export.
            Can be a pretrained model name (e.g., "fai-detr-m-coco") or
            path to a local model checkpoint.
        format (Optional[ExportFormat], optional): Target export format.
            - ExportFormat.ONNX: Export to ONNX format
            - ExportFormat.TORCHSCRIPT: Export to TorchScript format
            Defaults to ExportFormat.TORCHSCRIPT.
        output_dir (Optional[str], optional): Directory to save the exported model.
            If None, uses the default export directory based on model name.
            Defaults to None.
        device (Optional[Literal["cuda", "cpu"]], optional): Device to use during
            export process. Affects optimization and runtime selection.
            - "cuda": GPU-optimized export (requires CUDA)
            - "cpu": CPU-optimized export
            Defaults to "cuda" if None.
        onnx_opset (Optional[int], optional): ONNX opset version for ONNX exports.
            Higher versions support more operations but may have compatibility issues.
            Common versions: 11, 13, 16, 17. Defaults to 17 if None.
        im_size (Optional[Union[int, Tuple[int, int]]], optional): Input image size for the exported model.
            If int, treated as square (size, size). If tuple, treated as (height, width).
            Used to define fixed input shapes for optimization.
            If None, uses the model's default input size.
            Defaults to None.
        overwrite (Optional[bool], optional): Whether to overwrite existing exported
            files. If False, export will fail if target file already exists.
            Defaults to False if None.

    Raises:
        Exception: If model loading fails, export process encounters an error,
            or output file cannot be written.
        FileNotFoundError: If specified model name/path cannot be found.
        FileExistsError: If output file exists and overwrite=False.
        RuntimeError: If CUDA device is specified but not available.

    Examples:
        Basic ONNX export:
        ```python
        export_command("fai-detr-m-coco")
        ```

        TorchScript export with custom configuration:
        ```python
        export_command(model_name="fai-detr-m-coco", format=ExportFormat.TORCHSCRIPT, device="cuda", im_size=1024, overwrite=True)
        ```

        ONNX export for edge deployment:
        ```python
        export_command(model_name="my-model", format=ExportFormat.ONNX, device="cpu", onnx_opset=11, output_dir="./edge_models")
        ```

    Returns:
        str: Path to the exported model file.

    Note:
        - ONNX exports may take longer due to graph optimization
        - TorchScript exports maintain full PyTorch compatibility
        - Consider target deployment environment when choosing format
        - Test exported models before production deployment

    See Also:
        - [`focoos.model_manager.ModelManager.get`][focoos.model_manager.ModelManager.get]: Model loading
        - [`focoos.models.base_model.BaseModel.export`][focoos.models.base_model.BaseModel.export]: Core export method
        - [`focoos.ports.ExportFormat`][focoos.ports.ExportFormat]: Export format enumeration
    """
    # Load model
    logger.info(f"🔄 Loading model: {model_name}")
    model = ModelManager.get(name=model_name)

    # Mapping default runtime based on export format
    if format == ExportFormat.ONNX:
        runtime_type = RuntimeType.ONNX_CUDA32 if (device or "cuda") == "cuda" else RuntimeType.ONNX_CPU
    else:
        runtime_type = RuntimeType.TORCHSCRIPT_32

    # Export model to the specified format
    logger.info(f"🚀 Exporting to {format.value if format else 'torchscript'} format...")
    logger.info("📋 Configuration:")
    logger.info(f"  🎯 Runtime: {runtime_type}")
    logger.info(f"  💻 Device: {device or 'cuda'}")
    logger.info(f"  📐 Image size: {im_size or 'default'}")
    logger.info(f"  🔧 ONNX opset: {onnx_opset or 17}")

    export_path = model.export(
        runtime_type=runtime_type,
        onnx_opset=onnx_opset or 17,
        out_dir=output_dir,
        device=device or "cuda",
        overwrite=overwrite or False,
        image_size=im_size,
    )

    logger.info("✅ Model exported successfully!")
    logger.info(f"📁 Export path: {export_path}")
    logger.info(f"🎯 Runtime type: {runtime_type}")

    return export_path
