"""Training command implementation.

This module implements the training command for the Focoos CLI. It provides
comprehensive functionality to train computer vision models on various datasets
with extensive configuration options including distributed training, mixed precision,
advanced optimization strategies, and monitoring capabilities.

**Key Features:**
- **Multi-Task Support**: Object detection, segmentation, and classification
- **Distributed Training**: Multi-GPU training with DDP support
- **Mixed Precision**: Automatic mixed precision for faster training
- **Advanced Optimization**: EMA, gradient clipping, and flexible schedulers
- **Monitoring**: Real-time metrics, visualization, and Hub synchronization
- **Flexibility**: Extensive hyperparameter configuration options

**Training Pipeline:**
1. **Model Loading**: Loads pretrained or custom model architectures
2. **Dataset Preparation**: Automatic dataset loading and augmentation
3. **Training Loop**: Optimized training with validation and checkpointing
4. **Monitoring**: Metrics tracking, visualization, and progress reporting
5. **Model Saving**: Checkpoint management and final model export

**Supported Optimizers:**
- **AdamW**: Default optimizer with weight decay
- **SGD**: Stochastic gradient descent with momentum
- **RMSprop**: Root mean square propagation

**Supported Schedulers:**
- **MultiStep**: Step-wise learning rate reduction
- **Cosine**: Cosine annealing schedule
- **Polynomial**: Polynomial decay schedule
- **Fixed**: Constant learning rate

**Advanced Features:**
- **Early Stopping**: Automatic training termination on plateau
- **EMA**: Exponential moving average for model weights
- **Gradient Clipping**: Prevents gradient explosion
- **Mixed Precision**: FP16 training for speed and memory efficiency
- **Distributed Training**: Efficient multi-GPU training

Examples:
    Basic training:
    ```bash
    focoos train --model fai-detr-m-coco --dataset mydataset.zip
    ```

    Advanced training with custom settings:
    ```bash
    focoos train --model fai-detr-m-coco --dataset mydataset.zip --max-iters 5000 --batch-size 8 --learning-rate 0.001 --early-stop --ema-enabled
    ```

    Multi-GPU training:
    ```bash
    focoos train --model fai-detr-m-coco --dataset mydataset.zip --num-gpus 4 --batch-size 32
    ```

    Programmatic usage:
    ```python
    from focoos.cli.commands import train_command

    train_command(
        model_name="fai-detr-m-coco",
        dataset_name="mydataset.zip",
        im_size=640,
        max_iters=3000,
        batch_size=16
    )
    ```

See Also:
    - [`focoos.model_manager.ModelManager`][focoos.model_manager.ModelManager]: Model loading and management
    - [`focoos.data.auto_dataset.AutoDataset`][focoos.data.auto_dataset.AutoDataset]: Dataset handling
    - [`focoos.trainer.trainer.Trainer`][focoos.trainer.trainer.Trainer]: Core training functionality
    - [`focoos.ports.TrainerArgs`][focoos.ports.TrainerArgs]: Training configuration
"""

from typing import Optional, Tuple, Union

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
    im_size: Union[int, Tuple[int, int]],
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
):
    """Train a computer vision model on a specified dataset.

    Loads a model and dataset, then trains the model using comprehensive training
    configuration with support for distributed training, mixed precision, advanced
    optimization strategies, and extensive monitoring capabilities.

    **Training Process:**
    1. **Initialization**: Model and dataset loading with validation
    2. **Augmentation**: Automatic data augmentation based on task type
    3. **Training Loop**: Iterative training with validation and checkpointing
    4. **Monitoring**: Real-time metrics, visualization, and progress tracking
    5. **Completion**: Final model saving and training summary

    **Optimization Features:**
    - **Mixed Precision**: Automatic FP16 training for speed and memory efficiency
    - **Gradient Clipping**: Prevents gradient explosion during training
    - **EMA**: Exponential moving average for model weight stabilization
    - **Learning Rate Scheduling**: Flexible LR decay strategies
    - **Early Stopping**: Automatic termination on validation plateau

    **Distributed Training:**
    - **Multi-GPU**: Efficient training across multiple GPUs
    - **DDP**: Distributed Data Parallel for large-scale training
    - **Buffer Management**: Configurable buffer broadcasting and unused parameter detection

    Args:
        model_name (str): Name or identifier of the model to train.
            Can be a pretrained model name (e.g., "fai-detr-m-coco") or
            path to a custom model configuration.
        dataset_name (str): Name or path of the dataset to use for training.
            Supports various formats including ZIP archives, directories,
            and dataset identifiers.
        dataset_layout (DatasetLayout): Layout format of the dataset.
            Supported formats: ROBOFLOW_COCO, YOLO, COCO, etc.
        im_size (Union[int, Tuple[int, int]]): Input image size for training.
            If int, treated as square (size, size). If tuple, treated as (height, width).
            Images are resized to this size while maintaining aspect ratio.
        run_name (str): Unique name for this training run. Used for
            experiment tracking, logging, and output organization.
        output_dir (Optional[str], optional): Directory to save training outputs
            including checkpoints, logs, and final models.
            Defaults to MODELS_DIR if None.
        ckpt_dir (Optional[str], optional): Custom directory for checkpoint storage.
            If None, uses output_dir/checkpoints. Defaults to None.
        init_checkpoint (Optional[str], optional): Path to initial checkpoint for
            fine-tuning or transfer learning. Defaults to None.
        resume (bool, optional): Whether to resume training from the last checkpoint.
            Automatically finds the latest checkpoint if True. Defaults to False.
        num_gpus (int, optional): Number of GPUs to use for training.
            Enables distributed training if > 1. Defaults to available GPU count.
        device (DeviceType, optional): Device type for training.
            Options: "cuda", "cpu", "mps". Defaults to "cuda".
        workers (int, optional): Number of data loading workers for parallel
            data preprocessing. Higher values improve training speed.
            Defaults to 4.
        amp_enabled (bool, optional): Enable Automatic Mixed Precision training
            for faster training and reduced memory usage. Defaults to True.
        ddp_broadcast_buffers (bool, optional): Broadcast model buffers in
            Distributed Data Parallel mode. Defaults to False.
        ddp_find_unused (bool, optional): Find unused parameters in DDP mode
            for gradient computation optimization. Defaults to True.
        checkpointer_period (int, optional): Checkpoint saving frequency in iterations.
            Lower values save more frequently but use more storage. Defaults to 1000.
        checkpointer_max_to_keep (int, optional): Maximum number of checkpoints to keep.
            Older checkpoints are automatically deleted. Defaults to 1.
        eval_period (int, optional): Evaluation frequency in iterations.
            Determines how often validation is performed. Defaults to 50.
        log_period (int, optional): Logging frequency in iterations.
            Controls training progress output frequency. Defaults to 20.
        samples (int, optional): Number of samples to visualize during training
            for monitoring and debugging purposes. Defaults to 9.
        seed (int, optional): Random seed for reproducibility across runs.
            Ensures deterministic training behavior. Defaults to 42.
        early_stop (bool, optional): Enable early stopping to prevent overfitting.
            Monitors validation metrics and stops training on plateau. Defaults to True.
        patience (int, optional): Early stopping patience in evaluation periods.
            Number of evaluations without improvement before stopping. Defaults to 10.
        ema_enabled (bool, optional): Enable Exponential Moving Average for model weights.
            Provides more stable model performance. Defaults to False.
        ema_decay (float, optional): EMA decay rate for weight averaging.
            Higher values give more weight to recent updates. Defaults to 0.999.
        ema_warmup (int, optional): EMA warmup iterations before full application.
            Prevents instability in early training. Defaults to 2000.
        learning_rate (float, optional): Initial learning rate for optimization.
            Base learning rate before scheduler application. Defaults to 5e-4.
        weight_decay (float, optional): Weight decay for L2 regularization.
            Helps prevent overfitting by penalizing large weights. Defaults to 0.02.
        max_iters (int, optional): Maximum training iterations.
            Total number of training steps to perform. Defaults to 3000.
        batch_size (int, optional): Training batch size per GPU.
            Larger batches improve training stability but require more memory.
            Defaults to 16.
        scheduler (SchedulerType, optional): Learning rate scheduler type.
            Options: "MULTISTEP", "COSINE", "POLY", "FIXED". Defaults to "MULTISTEP".
        optimizer (OptimizerType, optional): Optimizer type for training.
            Options: "ADAMW", "SGD", "RMSPROP". Defaults to "ADAMW".
        weight_decay_norm (float, optional): Weight decay for normalization layers.
            Separate weight decay for batch norm and layer norm. Defaults to 0.0.
        weight_decay_embed (float, optional): Weight decay for embedding layers.
            Separate weight decay for embedding weights. Defaults to 0.0.
        backbone_multiplier (float, optional): Learning rate multiplier for backbone.
            Allows different learning rates for different model parts. Defaults to 0.1.
        decoder_multiplier (float, optional): Learning rate multiplier for decoder.
            Fine-tunes decoder learning rate independently. Defaults to 1.0.
        head_multiplier (float, optional): Learning rate multiplier for classification head.
            Adjusts head learning rate for better convergence. Defaults to 1.0.
        freeze_bn (bool, optional): Freeze batch normalization layers during training.
            Useful for fine-tuning with small datasets. Defaults to False.
        clip_gradients (float, optional): Gradient clipping value to prevent explosion.
            Clips gradients to maximum norm value. Defaults to 0.1.
        size_divisibility (int, optional): Size divisibility constraint for input images.
            Ensures input dimensions are divisible by this value. Defaults to 0.
        gather_metric_period (int, optional): Metric gathering frequency for monitoring.
            Controls how often metrics are computed and logged. Defaults to 1.
        zero_grad_before_forward (bool, optional): Zero gradients before forward pass.
            Optimization for certain training scenarios. Defaults to False.
        sync_to_hub (bool, optional): Synchronize model and metrics to Focoos Hub
            for experiment tracking and sharing. Defaults to False.
        datasets_dir (Optional[str], optional): Custom datasets directory.
            Override default dataset search location. Defaults to DATASETS_DIR if None.

    Raises:
        Exception: If model loading, dataset loading, or training fails.
        FileNotFoundError: If specified model, dataset, or checkpoint cannot be found.
        ValueError: If invalid training parameters are provided.
        RuntimeError: If GPU training is requested but not available.

    Examples:
        Basic training:
        ```python
        train_command(model_name="fai-detr-m-coco", dataset_name="mydataset.zip", dataset_layout=DatasetLayout.ROBOFLOW_COCO, im_size=640, run_name="my_training_run")
        ```

        Advanced training with custom settings:
        ```python
        train_command(
            model_name="fai-detr-m-coco",
            dataset_name="mydataset.zip",
            dataset_layout=DatasetLayout.ROBOFLOW_COCO,
            im_size=640,
            run_name="advanced_training",
            max_iters=5000,
            batch_size=32,
            learning_rate=1e-3,
            early_stop=True,
            ema_enabled=True,
            num_gpus=2,
        )
        ```

        Fine-tuning with custom checkpoint:
        ```python
        train_command(
            model_name="fai-detr-m-coco",
            dataset_name="custom_dataset",
            dataset_layout=DatasetLayout.COCO,
            im_size=1024,
            run_name="fine_tuning",
            init_checkpoint="path/to/checkpoint.pth",
            learning_rate=1e-4,
            freeze_bn=True,
        )
        ```

    Note:
        - Training progress is logged to console and optionally to Hub
        - Validation is performed periodically based on eval_period
        - Checkpoints are saved automatically during training
        - Early stopping monitors validation metrics to prevent overfitting
        - Multi-GPU training requires CUDA and sufficient GPU memory

    See Also:
        - [`focoos.model_manager.ModelManager.get`][focoos.model_manager.ModelManager.get]: Model loading
        - [`focoos.data.auto_dataset.AutoDataset`][focoos.data.auto_dataset.AutoDataset]: Dataset handling
        - [`focoos.ports.TrainerArgs`][focoos.ports.TrainerArgs]: Training configuration
        - [`focoos.trainer.trainer.Trainer`][focoos.trainer.trainer.Trainer]: Core training loop
    """
    logger.info(f"🚀 Starting training: {model_name} with dataset: {dataset_name}")
    logger.info(f"📋 Configuration: {im_size}px, {max_iters} iterations, batch size {batch_size}")

    try:
        # Load model
        logger.info(f"🔄 Loading model: {model_name}")
        model = ModelManager.get(model_name)

        # Initialize dataset
        logger.info(f"🔄 Loading dataset: {dataset_name}")
        auto_dataset = AutoDataset(
            dataset_name=dataset_name, task=model.task, layout=dataset_layout, datasets_dir=datasets_dir or DATASETS_DIR
        )

        # Get augmentations
        logger.info("🎨 Setting up data augmentations")
        train_augs, val_augs = get_default_by_task(model.task, resolution=im_size or model.model_info.im_size)
        train_dataset = auto_dataset.get_split(augs=train_augs.get_augmentations(), split=DatasetSplitType.TRAIN)
        valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

        # Configure training arguments
        logger.info("⚙️ Configuring training parameters")
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
        logger.info("🏃 Starting training loop...")
        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(valid_dataset)}")

        model.train(trainer_args, train_dataset, valid_dataset)

        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Output directory: {trainer_args.output_dir}")
        logger.info(f"🏷️ Run name: {trainer_args.run_name}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
