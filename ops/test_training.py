import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from focoos.data.auto_dataset import AutoDataset
from focoos.data.default_aug import get_default_by_task
from focoos.hub.api_client import ApiClient
from focoos.model_manager import ModelManager
from focoos.ports import DATASETS_DIR, DatasetLayout, DatasetSplitType, RuntimeType, Task, TrainerArgs
from focoos.utils.logger import get_logger

logger = get_logger("TestTraning")

datasets = [
    "chess-coco-detection.zip",
    "fire-coco-instseg.zip",
    "balloons-coco-sem.zip",
]


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
        ds_name = "balloons-coco-sem.zip"
        layout = DatasetLayout.ROBOFLOW_SEG

    elif task == Task.DETECTION:
        ds_name = "chess-coco-detection.zip"
        layout = DatasetLayout.ROBOFLOW_COCO
    elif task == Task.INSTANCE_SEGMENTATION:
        ds_name = "fire-coco-instseg.zip"
        layout = DatasetLayout.ROBOFLOW_COCO
    elif task == Task.KEYPOINT:
        ds_name = "basket-court-keypoint.zip"
        layout = DatasetLayout.ROBOFLOW_COCO
    else:
        raise ValueError(f"Error: task {task} not supported")
    url = f"https://public.focoos.ai/datasets/{ds_name}"
    api_client = ApiClient()
    api_client.download_ext_file(url, DATASETS_DIR, skip_if_exists=True)
    return ds_name, layout


def train(model_name: str):
    model = ModelManager.get(model_name)

    # Convert string task to Task enum
    task = Task(model.model_info.task)

    dataset_name, layout = get_dataset(task)

    # Initialize dataset
    auto_dataset = AutoDataset(dataset_name=dataset_name, task=task, layout=layout)
    resolution = 640

    # Get default augmentations for the specified task
    train_augs, val_augs = get_default_by_task(task, resolution)

    train_augs.crop_size = resolution
    train_augs.crop = True

    train_dataset = auto_dataset.get_split(augs=train_augs.get_augmentations(), split=DatasetSplitType.TRAIN)
    valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

    # Get again the model with the correct number of classes
    model = ModelManager.get(model_name, num_classes=train_dataset.dataset.metadata.num_classes)

    _temp_dir = tempfile.mkdtemp()
    # out_dir = os.path.join(_temp_dir, "output")
    logger.info(f"Created temporary directory for training output: {_temp_dir}")

    # Configure training arguments
    trainer_args = TrainerArgs(
        run_name=model_name + "_test",
        # output_dir=out_dir,
        amp_enabled=True,
        batch_size=8,
        max_iters=50,
        eval_period=50,
        learning_rate=1e-4,
        scheduler="MULTISTEP",
        weight_decay=0.0,
        workers=4,
    )

    # Start training
    model.train(trainer_args, train_dataset, valid_dataset)
    infer = model.export(runtime_type=RuntimeType.ONNX_CUDA32, overwrite=True)
    infer.benchmark(iterations=50)
    infer = model.export(runtime_type=RuntimeType.TORCHSCRIPT_32, overwrite=True)
    infer.benchmark(iterations=50)

    out_dir = trainer_args.output_dir
    files = list_files_with_extensions_recursively(out_dir)
    files_to_check = ["log.txt", "model_final.pth", "model_info.json", "metrics.json", "model.onnx", "model.pt"]
    for file in files_to_check:
        assert any(os.path.basename(f) == file for f in files), f"File {file} not found in {out_dir}"

    print(f"✅ {model_name} TEST DONE, {files_to_check} correctly found in {out_dir}. ======================")


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="Train a pretrained model")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train")

    args = parser.parse_args()
    logger.info(f"🚀 Start training test: {args.model} =================================================")
    torch.cuda.empty_cache()
    train(args.model)
