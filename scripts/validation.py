"""
Model Validation Example

This script demonstrates how to validate a model using the FocoOS framework for various computer vision tasks.
It shows the workflow for evaluating a model on a validation dataset with customizable parameters.

Usage:
    python examples/validation.py --dataset_name coco_2017_instance --model_name fai-mf-s-coco-ins --task instance_segmentation --resolution 1024 --batch_size 16

Parameters:
    --dataset_name (str): Name of the dataset to use (default: "coco_2017_instance")
    --task (str): Task type, one of "detection", "segmentation", "instance_segmentation", "classification" (default: "instance_segmentation")
    --layout (str): Dataset layout format (default: "catalog")
    --model_name (str): Name of the model to use (default: "fai-mf-s-coco-ins")
    --resolution (int): Input resolution for validation (default: 1024)
    --batch_size (int): Batch size for validation (default: 16)
    --output_dir (str): Directory to save validation outputs (default: "./experiments")
    --run_name (str): Name for this validation run (default: None - uses model name)
    --workers (int): Number of data loading workers (default: 16)
"""

import argparse

from focoos.data.auto_dataset import AutoDataset
from focoos.data.default_aug import get_default_by_task
from focoos.model_manager import ModelManager
from focoos.ports import DatasetLayout, DatasetSplitType, Task, TrainerArgs


def parse_args():
    parser = argparse.ArgumentParser(description="Model Validation Example")
    parser.add_argument("--dataset_name", type=str, default="coco_2017_instance", help="Name of the dataset")
    parser.add_argument(
        "--task",
        type=str,
        default=Task.INSTANCE_SEGMENTATION.value,
        choices=[task.value for task in Task],
        help="Task type",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=DatasetLayout.CATALOG.value,
        choices=[layout.value for layout in DatasetLayout],
        help="Dataset layout format",
    )
    parser.add_argument("--model_name", type=str, default="fai-mf-s-coco-ins", help="Model name")
    parser.add_argument("--resolution", type=int, default=1024, help="Input resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (defaults to model name if None)")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--advanced_aug", action="store_true", help="Use advanced augmentations")
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert string task to Task enum
    task = Task(args.task)

    # Convert string layout to DatasetLayout enum
    layout = DatasetLayout(args.layout)

    # Initialize dataset
    auto_dataset = AutoDataset(dataset_name=args.dataset_name, task=task, layout=layout)
    resolution = args.resolution

    # Get default augmentations for the specified task
    train_augs, val_augs = get_default_by_task(task, resolution, advanced=args.advanced_aug)
    valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

    # Initialize model
    model = ModelManager.get(args.model_name)

    # Use model name as run name if not specified
    run_name = args.run_name if args.run_name is not None else model.model_info.name

    # Configure validation arguments
    trainer_args = TrainerArgs(
        run_name=run_name,
        output_dir=args.output_dir,
        amp_enabled=True,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    # Start validation
    model.eval(trainer_args, valid_dataset)


if __name__ == "__main__":
    main()
