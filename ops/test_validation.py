import os
import tempfile

from focoos.data.auto_dataset import AutoDataset
from focoos.data.default_aug import get_default_by_task
from focoos.model_manager import ModelManager
from focoos.ports import DATASETS_DIR, DatasetLayout, DatasetSplitType, Task, TrainerArgs
from focoos.utils.logger import get_logger

logger = get_logger("TestValidation")

THRESHOLD = 0.01  # 1% acceptable variation


def train(model_name: str):
    model = ModelManager.get(model_name)
    if model.model_info.val_metrics is None:
        raise ValueError(f"Model {model_name} has no validation metrics. Please run training first.")
    current_val_metrics = model.model_info.val_metrics.copy()

    # Convert string task to Task enum
    task = Task(model.model_info.task)

    dataset_name, layout = model.model_info.val_dataset, DatasetLayout.CATALOG
    assert dataset_name is not None, f"Dataset name is not set for model {model_name}"

    # Initialize dataset
    try:
        auto_dataset = AutoDataset(dataset_name=dataset_name, task=task, layout=layout, datasets_dir=DATASETS_DIR)
    except Exception as e:
        logger.error(f"Error initializing dataset: {e}. Check you have it downloaded and registered in the hub.")
        raise
    resolution = model.model_info.im_size

    # Get default augmentations for the specified task
    train_augs, val_augs = get_default_by_task(task, resolution)
    valid_dataset = auto_dataset.get_split(augs=val_augs, split=DatasetSplitType.VAL)

    _temp_dir = tempfile.mkdtemp()
    out_dir = os.path.join(_temp_dir, "output")
    logger.info(f"Created temporary directory for training output: {_temp_dir}")

    # Configure training arguments
    trainer_args = TrainerArgs(
        run_name=model_name + "_test",
        output_dir=out_dir,
        amp_enabled=True,
        batch_size=8,
        max_iters=100,
        eval_period=50,
        learning_rate=1e-4,
        scheduler="MULTISTEP",
        weight_decay=0.0,
        workers=4,
    )

    # Start training
    model.eval(trainer_args, valid_dataset)

    original_metrics = dict(model.model_info.val_metrics.items())
    diff = {k: abs(v - current_val_metrics[k]) for k, v in original_metrics.items() if v != current_val_metrics[k]}
    valid = True
    for k, v in diff.items():
        if v > (THRESHOLD * original_metrics[k]):
            logger.warning(f"{k}: {current_val_metrics[k]} -> {original_metrics[k]} ({v})")
            valid = False
    if valid:
        logger.info(f"✅ TEST DONE, Model {model_name} validated.")
    else:
        logger.warning(f"❌ TEST FAILED, Model {model_name} didn't validate.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a pretrained model")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to validate")

    args = parser.parse_args()
    logger.info(f"Training model: {args.model}")
    train(args.model)
