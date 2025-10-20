"""
Lightning-based training for Focoos models.
This module provides an alternative training approach using PyTorch Lightning.
"""

import os
from typing import Optional

import torch

from focoos.hub.focoos_hub import FocoosHUB
from focoos.models.base_model import BaseModelNN
from focoos.ports import ArtifactName, HubSyncLocalTraining, ModelInfo, ModelStatus, TrainArgs
from focoos.processor.base_processor import Processor
from focoos.utils.logger import get_logger

logger = get_logger("LightningTrainer")


def run_train_lightning(
    train_args: TrainArgs,
    datamodule=None,  # Optional[FocoosLightningDataModule]
    image_model: Optional[BaseModelNN] = None,
    processor: Optional[Processor] = None,
    model_info: Optional[ModelInfo] = None,
    hub: Optional[FocoosHUB] = None,
):
    """Run model training using PyTorch Lightning.

    This function provides an alternative training approach using PyTorch Lightning's
    Trainer instead of the custom training loop. It offers:
    - Simplified training logic with Lightning abstractions
    - Built-in support for callbacks, logging, and checkpointing
    - Better integration with modern ML workflows
    - Easier customization and extension

    Args:
        train_args: Extended training configuration (TrainArgs) with dataset parameters
        datamodule: Optional FocoosLightningDataModule. If not provided, will be created from train_args
        image_model: Model to train
        processor: Processor for data preprocessing
        model_info: Model metadata/configuration
        hub: Optional Focoos Hub instance for model syncing

    Returns:
        tuple: (trained model, updated metadata)

    Example:
        >>> from focoos.ports import TrainArgs, Task, DatasetLayout
        >>> from focoos.trainer.trainer_lightning import run_train_lightning
        >>> train_args = TrainArgs(
        ...     run_name="my_training",
        ...     dataset_name="coco_2017_det",
        ...     task=Task.DETECTION,
        ...     layout=DatasetLayout.CATALOG,
        ...     batch_size=8,
        ...     image_size=640,
        ...     max_iters=1000,
        ... )
        >>> run_train_lightning(
        ...     train_args=train_args,
        ...     image_model=model,
        ...     processor=processor,
        ...     model_info=model_info,
        ... )
    """
    try:
        import lightning as L
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import TensorBoardLogger
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required for this training mode. Install it with: pip install lightning"
        )

    from focoos.data.lightning import FocoosLightningDataModule
    from focoos.trainer.lightning_module import FocoosLightningModule

    logger.info("🚀 Starting Lightning training...")

    # Validate required parameters
    if image_model is None:
        raise ValueError("image_model is required")
    if processor is None:
        raise ValueError("processor is required")
    if model_info is None:
        raise ValueError("model_info is required")

    # Type narrowing for linter
    assert image_model is not None
    assert processor is not None
    assert model_info is not None

    # Create datamodule if not provided
    if datamodule is None:
        if not train_args.dataset_name:
            raise ValueError("dataset_name must be provided in train_args if datamodule is not provided")

        logger.info("📦 Creating DataModule from TrainArgs:")
        logger.info(f"   Dataset: {train_args.dataset_name}")
        logger.info(f"   Task: {train_args.task}")
        logger.info(f"   Layout: {train_args.layout}")
        logger.info(f"   Image size: {train_args.image_size}")

        datamodule = FocoosLightningDataModule(
            dataset_name=train_args.dataset_name,
            task=train_args.task,
            layout=train_args.layout,
            datasets_dir=train_args.datasets_dir,
            batch_size=train_args.batch_size,
            num_workers=train_args.workers,
            image_size=train_args.image_size,
            pin_memory=train_args.pin_memory,
            persistent_workers=train_args.persistent_workers,
        )

    # Setup output directory
    output_dir = os.path.join(train_args.output_dir, train_args.run_name.strip())
    os.makedirs(output_dir, exist_ok=True)

    # Create Lightning module
    lightning_module = FocoosLightningModule(
        model=image_model,
        processor=processor,
        model_info=model_info,
        train_args=train_args,
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback - save last model (monitoring disabled for now)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="model_best",
        save_last=True,
        save_top_k=0,  # Don't save top-k, just save last
    )
    callbacks.append(checkpoint_callback)

    # # Early stopping callback (temporarily disabled)
    # if train_args.early_stop:
    #     early_stop_callback = EarlyStopping(
    #         monitor=lightning_module.val_metric_key,
    #         patience=train_args.patience,
    #         mode="max",
    #         verbose=True,
    #     )
    #     callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="lightning_logs",
    )

    # Calculate max epochs from max_iters
    # Approximate epochs based on dataset size and batch size
    steps_per_epoch = len(datamodule.train_dataset) // train_args.batch_size
    max_epochs = max(1, train_args.max_iters // steps_per_epoch)

    logger.info(f"📊 Training for {max_epochs} epochs (~{train_args.max_iters} steps)")
    logger.info(f"📊 Dataset: {len(datamodule.train_dataset)} train samples, {len(datamodule.val_dataset)} val samples")

    # Create Lightning Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if train_args.device == "cuda" else "cpu",
        devices=train_args.num_gpus,
        callbacks=callbacks,
        logger=tb_logger,
        enable_checkpointing=True,
        log_every_n_steps=train_args.log_period,
        check_val_every_n_epoch=1,  # Validate every epoch
        num_sanity_val_steps=1,  # Run 1 sanity validation batch before training (default is 2)
        gradient_clip_val=train_args.clip_gradients if train_args.clip_gradients > 0 else None,
        precision="16-mixed" if train_args.amp_enabled else "32-true",
        deterministic=False,
        default_root_dir=output_dir,
    )

    # Train the model with the provided datamodule
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
    )

    # Save final model
    final_model_path = os.path.join(output_dir, "model_final.pth")
    torch.save(
        {
            "model": lightning_module.model.state_dict(),
        },
        final_model_path,
    )

    # Update model info
    model_info.weights_uri = final_model_path
    model_info.dump_json(os.path.join(output_dir, ArtifactName.INFO))

    logger.info(f"✅ Lightning training completed! Model saved to {final_model_path}")

    # Sync to hub if enabled
    if train_args.sync_to_hub and hub:
        logger.info("📤 Syncing to Focoos Hub...")
        remote_model = hub.new_model(model_info)
        if remote_model:
            remote_model.sync_local_training_job(
                local_training_info=HubSyncLocalTraining(
                    status=ModelStatus.TRAINING_COMPLETED,
                    iterations=train_args.max_iters,
                    training_info=model_info.training_info,
                ),
                dir=output_dir,
                upload_artifacts=[
                    ArtifactName.WEIGHTS,
                    ArtifactName.METRICS,
                ],
            )

    return lightning_module.get_model(), model_info
