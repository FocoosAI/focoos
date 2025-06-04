import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from focoos.infer.infer_model import InferModel
from focoos.model_manager import ModelManager
from focoos.ports import RuntimeType
from focoos.utils.logger import get_logger

logger = get_logger("TestExport")


def list_files_with_extensions_recursively(
    base_dir: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Generate a list of file paths with specific extensions recursively starting from a base directory.

    Parameters:
        base_directory (Union[str, Path]): The directory to start the recursive search from.
        extensions (Optional[List[str]]): List of file extensions to filter by. If None, all files will be included.

    Returns:
        List[Path]: A list of Path objects representing the file paths that match the criteria.
    """
    base_dir = Path(base_dir)
    file_paths = []

    if extensions:
        for extension in extensions:
            if extension.startswith("."):
                extension = extension[1:]
            _glob = f"*.{extension}"
            file_paths.extend(base_dir.rglob(_glob))
    else:
        file_paths.extend(base_dir.rglob("*"))

    return [path for path in file_paths if path.is_file()]


def generate_random_image(resolution: Tuple[int, int]) -> np.ndarray:
    """
    Generate a random image tensor with the specified resolution.

    Parameters:
        resolution (Tuple[int, int]): The height and width of the image.

    Returns:
        np.ndarray: A random image array with shape [height, width, 3].
    """
    height, width = resolution
    # Create a random RGB image (height, width, 3 channels)
    random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return random_image


def test_export(model_name: str):
    # Get the model from the registry
    model = ModelManager.get(model_name)
    logger.info(f"Loaded model: {model_name}")

    # Get default resolution from model info
    successful_exports = {}

    default_resolution = model.model_info.im_size
    logger.info(f"Default resolution: {default_resolution}")

    # Test with default resolution
    logger.info("Testing model with default resolution before export")
    # Generate input as tensor for original model
    input_tensor = torch.from_numpy(
        np.random.randint(0, 256, (1, 3, default_resolution, default_resolution), dtype=np.uint8)
    ).float()

    # Create numpy format for InferModel
    input_image = generate_random_image((default_resolution, default_resolution))

    try:
        with torch.no_grad():
            model(input_tensor)
        logger.info("Model successfully processed input before export")
        successful_exports["original"] = {"export": True, "resolutions": {default_resolution: True}}
        test_resolutions = [(224, 224), (320, 320), (480, 480), (640, 640), (512, 320)]

        for res in test_resolutions:
            if res == (default_resolution, default_resolution):
                continue

            logger.info(f"Testing model with resolution: {res}")
            test_image = generate_random_image(res)
            try:
                model(test_image)
                logger.info(f"Model successfully processed input with resolution {res}")
                successful_exports["original"]["resolutions"][res] = True
            except Exception as e:
                logger.warning(f"Error processing input with resolution {res}: {e}")
                successful_exports["original"]["resolutions"][res] = False
    except Exception as e:
        logger.error(f"Error processing input before export: {e}")
        raise

    # Create a temporary directory for exporting
    temp_dir = tempfile.mkdtemp()
    export_path = os.path.join(temp_dir, f"{model_name}_exported")
    logger.info(f"Created temporary directory for export: {export_path}")

    # Export the model
    for runtime_type in [v for v in RuntimeType if v != RuntimeType.ONNX_CUDA32]:
        try:
            model.export(runtime_type=runtime_type, out_dir=export_path)
            logger.info(f"Model successfully exported to: {export_path}")
            files = list_files_with_extensions_recursively(export_path)
            logger.info(f"Files in export directory: {[str(f) for f in files]}")

            # Expected files may vary based on export format, adjust as needed
            expected_extensions = [".onnx", ".json", ".pt"]
            for ext in expected_extensions:
                if not any(str(f).endswith(ext) for f in files):
                    logger.warning(f"No file with extension {ext} found in export directory")
            successful_exports[runtime_type] = {"export": True}
        except Exception as e:
            successful_exports[runtime_type] = {"export": False}
            logger.warning(f"Error exporting model: {e}")

        # Verify export files
        if successful_exports[runtime_type]["export"]:
            # Load the exported model
            exported_model = InferModel(export_path, model_info=model.model_info, runtime_type=runtime_type)
            logger.info("Successfully loaded exported model")
            successful_exports[runtime_type]["resolutions"] = {}

            # Test with default resolution on exported model
            logger.info("Testing exported model with default resolution")
            try:
                exported_model(input_image)
                logger.info("Exported model successfully processed input with default resolution")
                successful_exports[runtime_type]["resolutions"][default_resolution] = True
            except Exception as e:
                logger.error(f"Error processing input with default resolution after export: {e}")
                raise

            # Test with different resolutions
            test_resolutions = [(224, 224), (320, 320), (480, 480), (640, 640), (512, 320)]
            for res in test_resolutions:
                if res == (default_resolution, default_resolution):
                    continue

                logger.info(f"Testing exported model with resolution: {res}")
                test_image = generate_random_image(res)
                try:
                    exported_model(test_image)
                    logger.info(f"Exported model successfully processed input with resolution {res}")
                    successful_exports[runtime_type]["resolutions"][res] = True
                except Exception as e:
                    logger.warning(f"Error processing input with resolution {res}: {e}")
                    successful_exports[runtime_type]["resolutions"][res] = False

            latency = exported_model.benchmark()
            logger.info(f"============ model {model_name} {runtime_type} =============")
            logger.info(f"benchmark results: {latency}")
            logger.info(f"===========================================================")

    for runtime_type in successful_exports:
        if successful_exports[runtime_type]["export"]:
            logger.info(f"✅ EXPORT TEST DONE, model {model_name} successfully exported and tested.")
            for res, success in successful_exports[runtime_type]["resolutions"].items():
                logger.info(f"\t Resolution {res}: {'✅' if success else '❌'}")
        else:
            logger.info(f"❌ EXPORT TEST FAILED, model {model_name} failed to export.")
    return export_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a pretrained model and test it with various image resolutions")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to export and test")

    args = parser.parse_args()
    logger.info(f"Exporting and testing model: {args.model}")
    export_path = test_export(args.model)
    logger.info(f"Model exported to: {export_path}")
