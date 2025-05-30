"""
Example of using the classification dataset and mapper.

This script demonstrates how to:
1. Load a classification dataset from a folder structure
2. Apply augmentations using the ClassificationDatasetMapper
3. Visualize a few examples from the dataset

Usage:
    python examples/classification_dataset_example.py --data_dir /path/to/image_folder

The data_dir should have the following structure:
    /path/to/image_folder/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   ├── class2/
    │   │   ├── img3.jpg
    │   │   ├── img4.jpg
    ├── val/
    │   ├── class1/
    │   │   ├── img5.jpg
    │   ├── class2/
    │   │   ├── img6.jpg
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from focoos.data.datasets.dict_dataset import DictDataset
from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetMapper
from focoos.data.transforms import augmentation as A
from focoos.ports import DatasetSplitType


def parse_args():
    parser = argparse.ArgumentParser(description="Classification dataset example")
    parser.add_argument("--data_dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples")
    return parser.parse_args()


def get_train_transforms(crop_size=224):
    """Get standard training augmentations for classification"""
    return [
        A.RandomBrightness(0.9, 1.1),
        A.RandomContrast(0.9, 1.1),
        A.RandomSaturation(0.9, 1.1),
        A.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        A.ResizeShortestEdge(
            short_edge_length=[crop_size, int(crop_size * 1.5)],
            max_size=crop_size * 2,
            sample_style="choice",
        ),
        A.RandomCrop(crop_type="absolute", crop_size=(crop_size, crop_size)),
    ]


def get_val_transforms(crop_size=224):
    """Get standard validation transforms for classification"""
    return [
        A.ResizeShortestEdge(
            short_edge_length=crop_size,
            max_size=crop_size * 2,
        ),
        A.RandomCrop(crop_type="absolute", crop_size=(crop_size, crop_size)),
    ]


def visualize_batch(batch, dataset):
    """Visualize a batch of images with their labels"""
    # Get class names from the dataset metadata
    class_names = dataset.metadata.thing_classes

    # Create a figure with subplots
    batch_size = len(batch)
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 4, 4))

    for i, data in enumerate(batch):
        # Get the image and label
        image = data.image
        label = data.label

        # Convert tensor to numpy array and transpose from (C,H,W) to (H,W,C)
        image_np = image.numpy().transpose(1, 2, 0)

        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)

        # Get the class name
        class_name = class_names[label] if label is not None else "Unknown"

        # Plot the image
        if batch_size == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(image_np)
        ax.set_title(f"Class: {class_name}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load the classification dataset
    train_dataset = DictDataset.from_folder(args.data_dir, split=DatasetSplitType.TRAIN)

    val_dataset = DictDataset.from_folder(args.data_dir, split=DatasetSplitType.VAL)

    print(f"Loaded training dataset with {len(train_dataset)} images")
    print(f"Loaded validation dataset with {len(val_dataset)} images")
    print(f"Classes: {train_dataset.metadata.thing_classes}")

    # Create the dataset mappers with augmentations
    train_mapper = ClassificationDatasetMapper(
        is_train=True,
        augmentations=get_train_transforms(),
    )

    val_mapper = ClassificationDatasetMapper(
        is_train=False,
        augmentations=get_val_transforms(),
    )

    # Function to apply the mapper to each dataset element
    def map_dataset_element(dataset_dict):
        return train_mapper(dataset_dict)

    def map_val_dataset_element(dataset_dict):
        return val_mapper(dataset_dict)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: [map_dataset_element(x_i) for x_i in x],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: [map_val_dataset_element(x_i) for x_i in x],
    )

    # Visualize a batch if requested
    if args.visualize:
        print("Visualizing a batch from the training set...")
        for batch in train_loader:
            visualize_batch(batch, train_dataset)
            break

        print("Visualizing a batch from the validation set...")
        for batch in val_loader:
            visualize_batch(batch, val_dataset)
            break

    print("Example usage complete!")


if __name__ == "__main__":
    main()
