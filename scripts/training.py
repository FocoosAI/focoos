"""
Model Training Example

This script demonstrates how to train a model using the FocoOS framework for various computer vision tasks.
It shows the complete workflow from dataset loading to model training with customizable parameters.

Usage:
    python examples/training.py --dataset_name aquarium --model_name fai-detr-m-coco --task detection --resolution 640 --batch_size 16

Parameters:
    --dataset_name (str): Name of the dataset to use (default: "aquarium")
    --task (str): Task type, one of "detection", "segmentation", "classification" (default: "detection")
    --layout (str): Dataset layout format (default: "roboflow_coco")
    --model_name (str): Name of the model to use (default: "fai-detr-m-coco")
    --resolution (int): Input resolution for training (default: 640)
    --batch_size (int): Batch size for training (default: 16)
    --max_iters (int): Maximum number of training iterations (default: 100)
    --learning_rate (float): Learning rate for training (default: 0.0001)
    --output_dir (str): Directory to save training outputs (default: "./experiments")
    --run_name (str): Name for this training run (default: "exp1")
    --workers (int): Number of data loading workers (default: 16)
    --advanced_aug (bool): Whether to use advanced augmentations (default: False)
"""

import argparse

from focoos.data.auto_dataset import AutoDataset
from focoos.data.default_aug import get_default_by_task
from focoos.model_manager import ModelManager
from focoos.ports import DatasetLayout, DatasetSplitType, Task, TrainerArgs


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Example")
    parser.add_argument("--dataset_name", type=str, default="aquarium", help="Name of the dataset")
    parser.add_argument(
        "--task",
        type=str,
        default=Task.DETECTION.value,
        choices=[task.value for task in Task],
        help="Task type",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=DatasetLayout.ROBOFLOW_COCO.value,
        choices=[layout.value for layout in DatasetLayout],
        help="Dataset layout format",
    )
    parser.add_argument("--model_name", type=str, default="fai-detr-m-coco", help="Model name")
    parser.add_argument("--resolution", type=int, default=640, help="Input resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum training iterations")
    parser.add_argument("--eval_period", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Output directory")
    parser.add_argument("--run_name", type=str, default="exp1", help="Run name")
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
    train_dataset = auto_dataset.get_split(augs=train_augs.get_augmentations(), split=DatasetSplitType.TRAIN)
    valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

    # Initialize model
    model = ModelManager.get(args.model_name, num_classes=train_dataset.dataset.metadata.num_classes)

    # Configure training arguments
    trainer_args = TrainerArgs(
        run_name=args.run_name,
        output_dir=args.output_dir,
        amp_enabled=True,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_period=args.eval_period,
        learning_rate=args.learning_rate,
        scheduler="MULTISTEP",
        weight_decay=args.weight_decay,
        workers=args.workers,
    )

    # Start training
    model.train(trainer_args, train_dataset, valid_dataset)


if __name__ == "__main__":
    main()
