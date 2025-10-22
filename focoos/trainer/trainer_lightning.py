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
from focoos.utils.distributed.dist import comm
from focoos.utils.env import seed_all_rng
from focoos.utils.logger import capture_all_output, get_logger

try:
    import lightning as L
    from lightning.pytorch.callbacks import (
        BatchSizeFinder,
        DeviceStatsMonitor,
        LearningRateMonitor,
        ModelCheckpoint,
        RichModelSummary,
        ThroughputMonitor,
        Timer,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    raise ImportError("PyTorch Lightning is required for this training mode. Install it with: pip install lightning")

from focoos.data.lightning import FocoosLightningDataModule
from focoos.trainer.callbacks import MetricsJSONWriter, VisualizationCallback
from focoos.trainer.lightning_module import FocoosLightningModule

logger = get_logger("LightningTrainer")


def run_train_lightning(
    train_args: TrainArgs,
    image_model: Optional[BaseModelNN] = None,
    processor: Optional[Processor] = None,
    model_info: Optional[ModelInfo] = None,
    hub: Optional[FocoosHUB] = None,
    eval_only: bool = False,
):
    """Run model training or evaluation using PyTorch Lightning.

    This function provides an alternative training approach using PyTorch Lightning's
    Trainer instead of the custom training loop. It offers:
    - Simplified training logic with Lightning abstractions
    - Built-in support for callbacks, logging, and checkpointing
    - Better integration with modern ML workflows
    - Easier customization and extension

    Args:
        train_args: Extended training configuration (TrainArgs) with dataset parameters
        image_model: Model to train or evaluate
        processor: Processor for data preprocessing
        model_info: Model metadata/configuration
        hub: Optional Focoos Hub instance for model syncing
        eval_only: If True, only run validation without training

    Returns:
        tuple: (trained model, updated metadata) or validation results if eval_only=True

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

    # Validate required parameters
    if image_model is None:
        raise ValueError("image_model is required")
    if processor is None:
        raise ValueError("processor is required")
    if model_info is None:
        raise ValueError("model_info is required")
    if not train_args.dataset_name:
        raise ValueError("dataset_name must be provided in train_args")

    # Type narrowing for linter
    assert image_model is not None
    assert processor is not None
    assert model_info is not None

    # Set random seed for reproducibility
    seed_all_rng(None if train_args.seed < 0 else train_args.seed + comm.get_rank())

    # Create datamodule from train_args (detailed info logged by datamodule itself)

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
        seed=train_args.seed,
    )

    # Prepare model for training (similar to old trainer)
    from focoos.nn.layers.norm import FrozenBatchNorm2d
    from focoos.trainer.solver import ema

    # Set processor to train mode
    processor = processor.train()

    # Apply model modifications
    if train_args.freeze_bn:
        image_model = FrozenBatchNorm2d.convert_frozen_batchnorm(image_model)

    # Setup SyncBatchNorm for multi-GPU if needed
    if comm.get_world_size() > 1:
        image_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(image_model)

    # Move model to device
    image_model = image_model.to(train_args.device)

    # Setup EMA if enabled
    if train_args.ema_enabled and not hasattr(image_model, "ema_state"):
        ema.build_model_ema(image_model)

    # Set model to appropriate mode
    if eval_only:
        image_model = image_model.eval()
    else:
        image_model = image_model.train()

    # Setup output directory
    output_dir = os.path.join(train_args.output_dir, train_args.run_name)
    revision = 1
    original_run_name = train_args.run_name
    while os.path.exists(output_dir):
        train_args.run_name = original_run_name + f"-{revision}"
        output_dir = os.path.join(train_args.output_dir, train_args.run_name)
        revision += 1
    os.makedirs(output_dir, exist_ok=True)

    with capture_all_output(log_path=os.path.join(output_dir, "log.txt"), rank=comm.get_local_rank()):
        # Create Lightning module
        lightning_module = FocoosLightningModule(
            model=image_model,
            processor=processor,
            model_info=model_info,
            train_args=train_args,
        )

        # Setup callbacks
        callbacks = []

        # Metrics JSON writer - save metrics to JSON file (compatible with old trainer)
        metrics_writer = MetricsJSONWriter(output_dir=output_dir, log_period=train_args.log_period)
        callbacks.append(metrics_writer)

        # Batch size finder - automatically find optimal batch size
        # Disabled by default, can be enabled via train_args.auto_scale_batch_size = True
        if train_args.auto_scale_batch_size:
            batch_size_finder = BatchSizeFinder(mode="power", steps_per_trial=3)
            callbacks.append(batch_size_finder)
            logger.info("🔍 BatchSizeFinder enabled - will search for optimal batch size")

        # Checkpoint callback - save last model (monitoring disabled for now)
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="model_best",
            save_last=True,
            save_top_k=0,  # Don't save top-k, just save last
        )
        callbacks.append(checkpoint_callback)

        visualization_callback = VisualizationCallback(
            period=train_args.eval_period,
            n_sample=2,
            output_dir=output_dir,
            confidence_threshold=0.3,
        )
        callbacks.append(visualization_callback)

        # # Early stopping callback (temporarily disabled)
        # if train_args.early_stop:
        #     early_stop_callback = EarlyStopping(
        #         monitor=lightning_module.val_metric_key,
        #         patience=train_args.patience,
        #         mode="max",
        #         verbose=True,
        #     )
        #     callbacks.append(early_stop_callback)

        # callbacks.append(csv_logger)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Rich model summary
        rich_summary = RichModelSummary(max_depth=2)
        callbacks.append(rich_summary)

        # Throughput monitor
        throughput_monitor = ThroughputMonitor(batch_size_fn=lambda batch: train_args.batch_size)
        callbacks.append(throughput_monitor)

        # Timer callback
        timer = Timer()
        callbacks.append(timer)

        # Device stats monitor (GPU/CPU metrics)
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)

        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name="lightning_logs",
        )
        # csv_logger = CSVLogger(save_dir=output_dir, name="lightning_logs")

        # Calculate max epochs from max_iters
        # Approximate epochs based on dataset size and batch size
        steps_per_epoch = len(datamodule.train_dataset) // train_args.batch_size
        max_epochs = max(1, train_args.max_iters // steps_per_epoch)
        eval_epochs = max(1, train_args.eval_period // steps_per_epoch)

        logger.info(
            f"📊 Training for {max_epochs} epochs (~{train_args.max_iters} steps) validate every {eval_epochs} epochs"
        )
        logger.info(
            f"📊 Dataset: {len(datamodule.train_dataset)} train samples, {len(datamodule.val_dataset)} val samples"
        )
        logger.info(f"📊 Batch size: {train_args.batch_size}")
        logger.info(f"📊 Learning rate: {train_args.learning_rate}")
        logger.info(f"📊 Weight decay: {train_args.weight_decay}")
        logger.info(f"📊 EMA enabled: {train_args.ema_enabled}")
        logger.info(f"📊 AMP enabled: {train_args.amp_enabled}")
        logger.info(f"📊 Gradient clipping: {train_args.clip_gradients}")
        # Create Lightning Trainer
        trainer = L.Trainer(
            # max_steps=train_args.max_iters,
            max_epochs=max_epochs,
            enable_progress_bar=True,
            enable_model_summary=True,
            sync_batchnorm=True,
            # val_check_interval=eval_epochs,
            check_val_every_n_epoch=eval_epochs,
            benchmark=True,
            # val_check_interval=train_args.eval_period,  # Validate every eval_period steps
            accelerator="gpu" if train_args.device == "cuda" else "cpu",
            accumulate_grad_batches=1,
            devices=train_args.num_gpus,
            callbacks=callbacks,
            logger=[tb_logger],
            enable_checkpointing=True,
            log_every_n_steps=train_args.log_period,
            num_sanity_val_steps=0,  # Run 1 sanity validation batch before training (default is 2)
            gradient_clip_val=train_args.clip_gradients if train_args.clip_gradients > 0 else None,
            precision="16-mixed" if train_args.amp_enabled else "32-true",
            deterministic=False,
            default_root_dir=output_dir,
        )
        output_lines = [
            " 🚀 Starting training ",
            f" 📁 output_dir: {output_dir}",
            "========== 🔧 Main Hyperparameters 🔧 ==========",
            f" - max_iter: {train_args.max_iters}",
            f" - eval_period: {train_args.eval_period}",
            f" - batch_size: {train_args.batch_size}",
            f" - learning_rate: {train_args.learning_rate}",
            " - resolution: !TODO",
            f" - optimizer: {train_args.optimizer}",
            f" - scheduler: {train_args.scheduler}",
            f" - weight_decay: {train_args.weight_decay}",
            f" - ema_enabled: {train_args.ema_enabled}",
            f" - amp_enabled: {train_args.amp_enabled}",
            "================================================",
        ]
        logger.info("\n".join(output_lines))

        # Train or validate the model with the provided datamodule
        if eval_only:
            # Only run validation
            logger.info("🚀 Starting validation...")
            results = trainer.validate(
                model=lightning_module,
                datamodule=datamodule,
            )
            logger.info(f"✅ Validation completed! Results: {results}")

            # Update model info with validation metrics
            if results and len(results) > 0:
                val_results = results[0]
                model_info.val_metrics = dict(val_results)
                model_info.dump_json(os.path.join(output_dir, ArtifactName.INFO))

            return results
        else:
            # Run training
            trainer.fit(
                model=lightning_module,
                datamodule=datamodule,
            )

        # Save final model (only for training)
        final_model_path = os.path.join(output_dir, "model_final.pth")
        torch.save(
            {
                "model": lightning_module.model.state_dict(),
            },
            final_model_path,
        )

        # Update model info with best validation metrics from training
        try:
            from focoos.utils.metrics import parse_metrics

            parsed_metrics = parse_metrics(os.path.join(output_dir, "metrics.json"))
            if parsed_metrics.best_valid_metric:
                model_info.val_metrics = parsed_metrics.best_valid_metric
                logger.info(f"📊 Best validation metrics: {parsed_metrics.best_valid_metric}")
        except Exception as e:
            logger.warning(f"Could not parse metrics.json: {e}")
            # Use the best metric tracked during training
            if lightning_module.best_val_metric > 0:
                model_info.val_metrics = {lightning_module.val_metric_key: lightning_module.best_val_metric}

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
