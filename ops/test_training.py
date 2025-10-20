import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from focoos.data.lightning import FocoosLightningDataModule
from focoos.model_manager import ModelManager
from focoos.ports import DATASETS_DIR, DatasetLayout, RuntimeType, Task, TrainArgs
from focoos.utils.api_client import ApiClient
from focoos.utils.logger import get_logger

logger = get_logger("TestTraining")


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


def get_dataset(task: Task):
    if task == Task.SEMSEG:
        ds_file = "balloons-coco-sem.zip"
        layout = DatasetLayout.ROBOFLOW_SEG

    elif task in [Task.DETECTION, Task.CLASSIFICATION]:
        ds_file = "chess-coco-detection.zip"
        layout = DatasetLayout.ROBOFLOW_COCO
    elif task == Task.INSTANCE_SEGMENTATION:
        ds_file = "fire-coco-instseg.zip"
        layout = DatasetLayout.ROBOFLOW_COCO
    elif task == Task.KEYPOINT:
        ds_file = "basket-court-keypoint.zip"
        layout = DatasetLayout.ROBOFLOW_COCO
    else:
        raise ValueError(f"Error: task {task} not supported")

    # Download and extract the dataset
    url = f"https://public.focoos.ai/datasets/{ds_file}"
    api_client = ApiClient()
    api_client.download_ext_file(url, DATASETS_DIR, skip_if_exists=True)

    # Return dataset name without .zip extension
    ds_name = ds_file.replace(".zip", "")
    return ds_name, layout


def train(model_name: str, iter: int):
    # Get model
    model = ModelManager.get(model_name)

    # Type narrowing for linter
    assert model.model_info is not None

    # Convert string task to Task enum
    task = Task(model.model_info.task)

    dataset_name, layout = get_dataset(task)

    # Create a temporary datamodule to get metadata (num_classes)
    resolution = 640
    logger.info(f"Loading dataset {dataset_name} to get metadata...")

    temp_datamodule = FocoosLightningDataModule(
        dataset_name=dataset_name,
        task=task,
        layout=layout,
        datasets_dir=DATASETS_DIR,
        batch_size=8,
        num_workers=4,
        image_size=resolution,
    )

    # Get the number of classes from datamodule
    num_classes = len(temp_datamodule.train_dataset.dict_dataset.metadata.classes)

    # Get model with correct number of classes
    model = ModelManager.get(model_name, num_classes=num_classes)

    _temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for training output: {_temp_dir}")

    # Configure training arguments with dataset parameters
    train_args = TrainArgs(
        run_name=model_name + "_test",
        dataset_name=dataset_name,
        task=task,
        layout=layout,
        datasets_dir=DATASETS_DIR,
        image_size=resolution,
        amp_enabled=True,
        batch_size=8,
        max_iters=iter,
        learning_rate=1e-4,
        scheduler="MULTISTEP",
        weight_decay=0.0,
        workers=4,
        num_gpus=1,
        pin_memory=True,
        persistent_workers=True,
    )

    # Start Lightning training (datamodule will be created automatically from train_args)
    logger.info("🚀 Starting Lightning training...")
    model.train_lightning(train_args)

    # Export models
    logger.info("📦 Exporting models...")
    infer = model.export(runtime_type=RuntimeType.ONNX_CUDA32, overwrite=True)
    infer.benchmark(iterations=50)
    infer = model.export(runtime_type=RuntimeType.TORCHSCRIPT_32, overwrite=True)
    infer.benchmark(iterations=50)

    # Verify output files
    out_dir = train_args.output_dir
    files = list_files_with_extensions_recursively(out_dir)
    files_to_check = ["model_final.pth", "model_info.json", "model.onnx", "model.pt"]
    for file in files_to_check:
        assert any(os.path.basename(f) == file for f in files), f"File {file} not found in {out_dir}"

    print(f"✅ {model_name} LIGHTNING TEST DONE, {files_to_check} correctly found in {out_dir}. ======================")


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="Train a pretrained model")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train")
    parser.add_argument("--iter", type=int, default=50, help="Number of iterations")

    args = parser.parse_args()
    logger.info(f"🚀 Start training test: {args.model} =================================================")
    torch.cuda.empty_cache()
    train(args.model, args.iter)
