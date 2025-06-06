"""Train command implementation.

This module implements the training command for the Focoos CLI. It provides
functionality to train computer vision models on various datasets with
comprehensive configuration options including distributed training, mixed
precision, and advanced optimization strategies.
"""

from typing import Optional

from focoos.data.auto_dataset import AutoDataset
from focoos.data.default_aug import get_default_by_task
from focoos.model_manager import ModelManager
from focoos.ports import (
    DATASETS_DIR,
    MODELS_DIR,
    DatasetLayout,
    DatasetSplitType,
    DeviceType,
    OptimizerType,
    SchedulerType,
    TrainerArgs,
    get_gpus_count,
)
from focoos.utils.logger import get_logger

logger = get_logger("train")


def train_command(
    model_name: str,
    ##################
    ## Dataset args
    dataset_name: str,
    dataset_layout: DatasetLayout,
    im_size: int,
    ##################
    ## Training args
    run_name: str,
    output_dir: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    resume: bool = False,
    # Logistics params
    num_gpus: int = get_gpus_count(),
    device: DeviceType = "cuda",
    workers: int = 4,
    amp_enabled: bool = True,
    ddp_broadcast_buffers: bool = False,
    ddp_find_unused: bool = True,
    checkpointer_period: int = 1000,
    checkpointer_max_to_keep: int = 1,
    eval_period: int = 50,
    log_period: int = 20,
    samples: int = 9,
    seed: int = 42,
    early_stop: bool = True,
    patience: int = 10,
    # EMA
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup: int = 2000,
    # Hyperparameters
    learning_rate: float = 5e-4,
    weight_decay: float = 0.02,
    max_iters: int = 3000,
    batch_size: int = 16,
    scheduler: SchedulerType = "MULTISTEP",
    optimizer: OptimizerType = "ADAMW",
    weight_decay_norm: float = 0.0,
    weight_decay_embed: float = 0.0,
    backbone_multiplier: float = 0.1,
    decoder_multiplier: float = 1.0,
    head_multiplier: float = 1.0,
    freeze_bn: bool = False,
    clip_gradients: float = 0.1,
    size_divisibility: int = 0,
    # Training specific
    gather_metric_period: int = 1,
    zero_grad_before_forward: bool = False,
    sync_to_hub: bool = False,
    datasets_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
):
    """Train a computer vision model on a specified dataset.

    Loads a model and dataset, then trains the model using the specified training
    configuration. Supports distributed training, mixed precision, EMA, and various
    optimization strategies.

    Args:
        model_name (str): Name of the model to train.
        dataset_name (str): Name of the dataset to use for training.
        dataset_layout (DatasetLayout): Layout format of the dataset.
        im_size (int): Input image size for training.
        run_name (str): Unique name for this training run.
        output_dir (Optional[str], optional): Directory to save training outputs.
            Defaults to MODELS_DIR if None.
        ckpt_dir (Optional[str], optional): Directory to save checkpoints.
            Defaults to None.
        init_checkpoint (Optional[str], optional): Path to initial checkpoint for
            fine-tuning. Defaults to None.
        resume (bool, optional): Whether to resume training from the last checkpoint.
            Defaults to False.
        num_gpus (int, optional): Number of GPUs to use for training.
            Defaults to available GPU count.
        device (DeviceType, optional): Device type for training. Defaults to "cuda".
        workers (int, optional): Number of data loading workers. Defaults to 4.
        amp_enabled (bool, optional): Enable automatic mixed precision training.
            Defaults to True.
        ddp_broadcast_buffers (bool, optional): Broadcast buffers in DDP mode.
            Defaults to False.
        ddp_find_unused (bool, optional): Find unused parameters in DDP mode.
            Defaults to True.
        checkpointer_period (int, optional): Checkpoint saving frequency in iterations.
            Defaults to 1000.
        checkpointer_max_to_keep (int, optional): Maximum number of checkpoints to keep.
            Defaults to 1.
        eval_period (int, optional): Evaluation frequency in iterations. Defaults to 50.
        log_period (int, optional): Logging frequency in iterations. Defaults to 20.
        samples (int, optional): Number of samples to log during training. Defaults to 9.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        early_stop (bool, optional): Enable early stopping. Defaults to True.
        patience (int, optional): Early stopping patience in evaluations. Defaults to 10.
        ema_enabled (bool, optional): Enable Exponential Moving Average. Defaults to False.
        ema_decay (float, optional): EMA decay rate. Defaults to 0.999.
        ema_warmup (int, optional): EMA warmup iterations. Defaults to 2000.
        learning_rate (float, optional): Initial learning rate. Defaults to 5e-4.
        weight_decay (float, optional): Weight decay for regularization. Defaults to 0.02.
        max_iters (int, optional): Maximum training iterations. Defaults to 3000.
        batch_size (int, optional): Training batch size. Defaults to 16.
        scheduler (SchedulerType, optional): Learning rate scheduler type.
            Defaults to "MULTISTEP".
        optimizer (OptimizerType, optional): Optimizer type. Defaults to "ADAMW".
        weight_decay_norm (float, optional): Weight decay for normalization layers.
            Defaults to 0.0.
        weight_decay_embed (float, optional): Weight decay for embedding layers.
            Defaults to 0.0.
        backbone_multiplier (float, optional): Learning rate multiplier for backbone.
            Defaults to 0.1.
        decoder_multiplier (float, optional): Learning rate multiplier for decoder.
            Defaults to 1.0.
        head_multiplier (float, optional): Learning rate multiplier for head.
            Defaults to 1.0.
        freeze_bn (bool, optional): Freeze batch normalization layers. Defaults to False.
        clip_gradients (float, optional): Gradient clipping value. Defaults to 0.1.
        size_divisibility (int, optional): Size divisibility constraint. Defaults to 0.
        gather_metric_period (int, optional): Metric gathering frequency. Defaults to 1.
        zero_grad_before_forward (bool, optional): Zero gradients before forward pass.
            Defaults to False.
        sync_to_hub (bool, optional): Sync model to HuggingFace Hub. Defaults to False.
        datasets_dir (Optional[str], optional): Custom datasets directory.
            Defaults to DATASETS_DIR if None.

    Raises:
        Exception: If model loading, dataset loading, or training fails.
    """
    logger.info(f"Training model: {model_name} with dataset: {dataset_name} and image size: {im_size}")

    try:
        model = ModelManager.get(model_name, models_dir=models_dir)

        # Initialize dataset
        logger.info(f"Loading dataset: {dataset_name}")
        auto_dataset = AutoDataset(
            dataset_name=dataset_name, task=model.task, layout=dataset_layout, datasets_dir=datasets_dir or DATASETS_DIR
        )

        # Get augmentations
        train_augs, val_augs = get_default_by_task(model.task, resolution=im_size or model.model_info.im_size)
        train_dataset = auto_dataset.get_split(augs=train_augs.get_augmentations(), split=DatasetSplitType.TRAIN)
        valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

        trainer_args = TrainerArgs(
            run_name=run_name,
            output_dir=output_dir or MODELS_DIR,
            ckpt_dir=ckpt_dir,
            init_checkpoint=init_checkpoint,
            resume=resume,
            num_gpus=num_gpus,
            device=device,
            batch_size=batch_size,
            workers=workers,
            amp_enabled=amp_enabled,
            ddp_broadcast_buffers=ddp_broadcast_buffers,
            ddp_find_unused=ddp_find_unused,
            checkpointer_period=checkpointer_period,
            checkpointer_max_to_keep=checkpointer_max_to_keep,
            eval_period=eval_period,
            log_period=log_period,
            samples=samples,
            seed=seed,
            early_stop=early_stop,
            patience=patience,
            ema_enabled=ema_enabled,
            ema_decay=ema_decay,
            ema_warmup=ema_warmup,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_iters=max_iters,
            scheduler=scheduler,
            optimizer=optimizer,
            weight_decay_norm=weight_decay_norm,
            weight_decay_embed=weight_decay_embed,
            backbone_multiplier=backbone_multiplier,
            decoder_multiplier=decoder_multiplier,
            head_multiplier=head_multiplier,
            freeze_bn=freeze_bn,
            clip_gradients=clip_gradients,
            size_divisibility=size_divisibility,
            gather_metric_period=gather_metric_period,
            zero_grad_before_forward=zero_grad_before_forward,
            sync_to_hub=sync_to_hub,
        )

        # Start training
        logger.info("Starting training...")
        model.train(trainer_args, train_dataset, valid_dataset)
        logger.info("✅ Training completed!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
