"""Validation command implementation.

This module implements the validation command for the Focoos CLI. It provides
functionality to evaluate computer vision models on validation datasets,
reporting comprehensive metrics and performance statistics.
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
    OptimizerType,
    SchedulerType,
    TrainerArgs,
    get_gpus_count,
)
from focoos.utils.logger import get_logger

logger = get_logger("val")


def val_command(
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
    device: str = "cuda",
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
    datasets_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
):
    """Validate a computer vision model on a specified dataset.

    Loads a model and validation dataset, then evaluates the model's performance
    using the specified configuration. Reports validation metrics and saves results.

    Args:
        model_name (str): Name of the model to validate.
        dataset_name (str): Name of the dataset to use for validation.
        dataset_layout (DatasetLayout): Layout format of the dataset.
        im_size (int): Input image size for validation.
        run_name (str): Unique name for this validation run.
        output_dir (Optional[str], optional): Directory to save validation outputs.
            Defaults to MODELS_DIR if None.
        ckpt_dir (Optional[str], optional): Directory to save checkpoints.
            Defaults to None.
        init_checkpoint (Optional[str], optional): Path to checkpoint to validate.
            Defaults to None.
        resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
        num_gpus (int, optional): Number of GPUs to use. Defaults to available GPU count.
        device (str, optional): Device type for validation. Defaults to "cuda".
        workers (int, optional): Number of data loading workers. Defaults to 4.
        amp_enabled (bool, optional): Enable automatic mixed precision. Defaults to True.
        ddp_broadcast_buffers (bool, optional): Broadcast buffers in DDP mode.
            Defaults to False.
        ddp_find_unused (bool, optional): Find unused parameters in DDP mode.
            Defaults to True.
        checkpointer_period (int, optional): Checkpoint saving frequency. Defaults to 1000.
        checkpointer_max_to_keep (int, optional): Maximum checkpoints to keep.
            Defaults to 1.
        eval_period (int, optional): Evaluation frequency in iterations. Defaults to 50.
        log_period (int, optional): Logging frequency in iterations. Defaults to 20.
        samples (int, optional): Number of samples to log. Defaults to 9.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        early_stop (bool, optional): Enable early stopping. Defaults to True.
        patience (int, optional): Early stopping patience. Defaults to 10.
        ema_enabled (bool, optional): Enable Exponential Moving Average. Defaults to False.
        ema_decay (float, optional): EMA decay rate. Defaults to 0.999.
        ema_warmup (int, optional): EMA warmup iterations. Defaults to 2000.
        learning_rate (float, optional): Learning rate (for consistency). Defaults to 5e-4.
        weight_decay (float, optional): Weight decay value. Defaults to 0.02.
        max_iters (int, optional): Maximum iterations. Defaults to 3000.
        batch_size (int, optional): Validation batch size. Defaults to 16.
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
        datasets_dir (Optional[str], optional): Custom datasets directory.
            Defaults to DATASETS_DIR if None.

    Raises:
        Exception: If model loading, dataset loading, or validation fails.
    """

    # Log all arguments
    logger.info("Validation arguments:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Dataset layout: {dataset_layout}")
    logger.info(f"  Image size: {im_size}")
    logger.info(f"  Run name: {run_name}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Checkpoint dir: {ckpt_dir}")
    logger.info(f"  Init checkpoint: {init_checkpoint}")
    logger.info(f"  Resume: {resume}")
    logger.info(f"  Num GPUs: {num_gpus}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Max iterations: {max_iters}")
    logger.info(f"  Scheduler: {scheduler}")
    logger.info(f"  Optimizer: {optimizer}")

    try:
        logger.info(f"Loading model: {model_name}")
        model = ModelManager.get(model_name, models_dir=models_dir)

        logger.info(f"Loading dataset: {dataset_name}")
        auto_dataset = AutoDataset(
            dataset_name=dataset_name,
            task=model.model_info.task,
            layout=dataset_layout,
            datasets_dir=datasets_dir or DATASETS_DIR,
        )

        # Get augmentations
        _, val_augs = get_default_by_task(task=model.model_info.task, resolution=im_size or model.model_info.im_size)
        valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

        # Configure validation arguments
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
        )

        # Start validation
        logger.info("Starting validation...")
        model.test(trainer_args, valid_dataset)
        logger.info("✅ Validation completed!")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise
