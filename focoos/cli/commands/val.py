"""Validation command implementation.

This module implements the validation command for the Focoos CLI. It provides
comprehensive functionality to evaluate computer vision models on validation datasets,
reporting detailed metrics, performance statistics, and model quality assessments
with support for various evaluation protocols.

**Key Features:**
- **Comprehensive Evaluation**: Detailed metrics for all supported tasks
- **Multi-Task Support**: Object detection, segmentation, and classification
- **Flexible Configuration**: Extensive evaluation parameters and settings
- **Performance Analysis**: Speed and accuracy benchmarking
- **Result Visualization**: Sample predictions and performance charts
- **Distributed Evaluation**: Multi-GPU validation support

**Evaluation Metrics:**
- **Object Detection**: mAP, AP50, AP75, precision, recall
- **Segmentation**: IoU, Dice coefficient, pixel accuracy
- **Classification**: Top-1/Top-5 accuracy, F1-score, confusion matrix

**Use Cases:**
- Model performance assessment
- Hyperparameter validation
- Model comparison and selection
- Dataset quality evaluation
- Production readiness testing
- Benchmark evaluation

**Validation Process:**
1. **Model Loading**: Loads trained model with optional checkpoint
2. **Dataset Preparation**: Configures validation dataset and augmentations
3. **Evaluation Loop**: Runs inference on validation samples
4. **Metrics Computation**: Calculates comprehensive evaluation metrics
5. **Results Reporting**: Generates detailed performance reports

Examples:
    Basic validation:
    ```bash
    focoos val --model fai-detr-m-coco --dataset mydataset.zip
    ```

    Validation with specific checkpoint:
    ```bash
    focoos val --model fai-detr-m-coco --dataset mydataset.zip --init-checkpoint path/to/model.pth
    ```

    Multi-GPU validation:
    ```bash
    focoos val --model fai-detr-m-coco --dataset mydataset.zip --num-gpus 4 --batch-size 32
    ```

See Also:
    - [`focoos.model_manager.ModelManager`][focoos.model_manager.ModelManager]: Model loading and management
    - [`focoos.trainer.evaluation`][focoos.trainer.evaluation]: Evaluation metrics and protocols
    - [`focoos.data.auto_dataset.AutoDataset`][focoos.data.auto_dataset.AutoDataset]: Dataset handling
    - [`focoos.ports.TrainerArgs`][focoos.ports.TrainerArgs]: Configuration parameters
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
):
    """Validate a computer vision model on a specified dataset.

    Loads a model and validation dataset, then evaluates the model's performance
    using comprehensive metrics and analysis. Supports various evaluation protocols
    and provides detailed performance reports with visualization capabilities.

    **Validation Process:**
    1. **Model Loading**: Loads trained model with optional checkpoint
    2. **Dataset Preparation**: Configures validation dataset with augmentations
    3. **Evaluation Setup**: Initializes evaluation metrics and protocols
    4. **Inference Loop**: Runs model inference on validation samples
    5. **Metrics Computation**: Calculates comprehensive performance metrics
    6. **Results Reporting**: Generates detailed validation reports

    **Performance Metrics:**
    - **Detection Models**: mAP, AP50, AP75, precision, recall, F1-score
    - **Segmentation Models**: IoU, Dice coefficient, pixel accuracy
    - **Classification Models**: Top-1/Top-5 accuracy, confusion matrix

    **Advanced Features:**
    - **Multi-GPU Evaluation**: Distributed validation for faster processing
    - **Mixed Precision**: FP16 evaluation for memory efficiency
    - **Sample Visualization**: Visual analysis of model predictions
    - **Performance Profiling**: Speed and memory usage analysis

    Args:
        model_name (str): Name or identifier of the model to validate.
            Can be a pretrained model name (e.g., "fai-detr-m-coco") or
            path to a custom model configuration.
        dataset_name (str): Name or path of the dataset to use for validation.
            Supports various formats including ZIP archives, directories,
            and dataset identifiers.
        dataset_layout (DatasetLayout): Layout format of the dataset.
            Supported formats: ROBOFLOW_COCO, YOLO, COCO, etc.
        im_size (Union[int, Tuple[int, int]]): Input image size for validation.
            If int, treated as square (size, size). If tuple, treated as (height, width).
            Images are resized to this size while maintaining aspect ratio.
        run_name (str): Unique name for this validation run. Used for
            result organization, logging, and report generation.
        output_dir (Optional[str], optional): Directory to save validation outputs
            including metrics, reports, and visualization results.
            Defaults to MODELS_DIR if None.
        ckpt_dir (Optional[str], optional): Custom directory for checkpoint loading.
            If None, uses default checkpoint locations. Defaults to None.
        init_checkpoint (Optional[str], optional): Path to specific checkpoint to validate.
            If None, uses the latest or best available checkpoint. Defaults to None.
        resume (bool, optional): Whether to resume from checkpoint state.
            Mainly used for consistency with training parameters. Defaults to False.
        num_gpus (int, optional): Number of GPUs to use for validation.
            Enables distributed evaluation if > 1. Defaults to available GPU count.
        device (str, optional): Device type for validation.
            Options: "cuda", "cpu", "mps". Defaults to "cuda".
        workers (int, optional): Number of data loading workers for parallel
            data preprocessing. Higher values improve evaluation speed.
            Defaults to 4.
        amp_enabled (bool, optional): Enable Automatic Mixed Precision evaluation
            for faster validation and reduced memory usage. Defaults to True.
        ddp_broadcast_buffers (bool, optional): Broadcast model buffers in
            Distributed Data Parallel mode. Defaults to False.
        ddp_find_unused (bool, optional): Find unused parameters in DDP mode
            for gradient computation optimization. Defaults to True.
        checkpointer_period (int, optional): Checkpoint period (for consistency).
            Not directly used in validation. Defaults to 1000.
        checkpointer_max_to_keep (int, optional): Maximum checkpoints to keep.
            Used for checkpoint management consistency. Defaults to 1.
        eval_period (int, optional): Evaluation period for metrics computation.
            Controls frequency of metric updates. Defaults to 50.
        log_period (int, optional): Logging frequency in iterations.
            Controls validation progress output frequency. Defaults to 20.
        samples (int, optional): Number of samples to visualize during validation
            for qualitative analysis and debugging. Defaults to 9.
        seed (int, optional): Random seed for reproducible validation results.
            Ensures consistent evaluation across runs. Defaults to 42.
        early_stop (bool, optional): Early stopping configuration (for consistency).
            Not directly used in validation mode. Defaults to True.
        patience (int, optional): Early stopping patience (for consistency).
            Not directly used in validation mode. Defaults to 10.
        ema_enabled (bool, optional): Enable EMA model evaluation if available.
            Uses exponential moving average weights if present. Defaults to False.
        ema_decay (float, optional): EMA decay rate (for consistency).
            Not directly used in validation mode. Defaults to 0.999.
        ema_warmup (int, optional): EMA warmup iterations (for consistency).
            Not directly used in validation mode. Defaults to 2000.
        learning_rate (float, optional): Learning rate (for consistency).
            Not directly used in validation mode. Defaults to 5e-4.
        weight_decay (float, optional): Weight decay (for consistency).
            Not directly used in validation mode. Defaults to 0.02.
        max_iters (int, optional): Maximum iterations (for consistency).
            Not directly used in validation mode. Defaults to 3000.
        batch_size (int, optional): Validation batch size per GPU.
            Larger batches improve evaluation speed but require more memory.
            Defaults to 16.
        scheduler (SchedulerType, optional): Scheduler type (for consistency).
            Not directly used in validation mode. Defaults to "MULTISTEP".
        optimizer (OptimizerType, optional): Optimizer type (for consistency).
            Not directly used in validation mode. Defaults to "ADAMW".
        weight_decay_norm (float, optional): Weight decay for normalization layers.
            Used for consistency with training configuration. Defaults to 0.0.
        weight_decay_embed (float, optional): Weight decay for embedding layers.
            Used for consistency with training configuration. Defaults to 0.0.
        backbone_multiplier (float, optional): Backbone learning rate multiplier.
            Used for consistency with training configuration. Defaults to 0.1.
        decoder_multiplier (float, optional): Decoder learning rate multiplier.
            Used for consistency with training configuration. Defaults to 1.0.
        head_multiplier (float, optional): Head learning rate multiplier.
            Used for consistency with training configuration. Defaults to 1.0.
        freeze_bn (bool, optional): Freeze batch normalization during validation.
            Ensures consistent evaluation behavior. Defaults to False.
        clip_gradients (float, optional): Gradient clipping value (for consistency).
            Not directly used in validation mode. Defaults to 0.1.
        size_divisibility (int, optional): Size divisibility constraint for input images.
            Ensures input dimensions are divisible by this value. Defaults to 0.
        gather_metric_period (int, optional): Metric gathering frequency.
            Controls how often metrics are computed and aggregated. Defaults to 1.
        zero_grad_before_forward (bool, optional): Zero gradients before forward pass.
            Optimization for certain evaluation scenarios. Defaults to False.
        datasets_dir (Optional[str], optional): Custom datasets directory.
            Override default dataset search location. Defaults to DATASETS_DIR if None.

    Raises:
        Exception: If model loading, dataset loading, or validation fails.
        FileNotFoundError: If specified model, dataset, or checkpoint cannot be found.
        ValueError: If invalid validation parameters are provided.
        RuntimeError: If GPU validation is requested but not available.

    Examples:
        Basic validation:
        ```python
        val_command(model_name="fai-detr-m-coco", dataset_name="mydataset.zip", dataset_layout=DatasetLayout.ROBOFLOW_COCO, im_size=640, run_name="validation_run")
        ```

        Validation with specific checkpoint:
        ```python
        val_command(
            model_name="fai-detr-m-coco",
            dataset_name="mydataset.zip",
            dataset_layout=DatasetLayout.ROBOFLOW_COCO,
            im_size=640,
            run_name="checkpoint_validation",
            init_checkpoint="path/to/model.pth",
        )
        ```

        Multi-GPU validation:
        ```python
        val_command(
            model_name="fai-detr-m-coco",
            dataset_name="mydataset.zip",
            dataset_layout=DatasetLayout.ROBOFLOW_COCO,
            im_size=640,
            run_name="distributed_validation",
            num_gpus=4,
            batch_size=32,
        )
        ```

    Note:
        - Validation metrics are logged to console and saved to output directory
        - Sample visualizations are generated for qualitative analysis
        - Performance metrics vary by model task (detection, segmentation, classification)
        - Multi-GPU validation requires CUDA and sufficient GPU memory
        - Results include both quantitative metrics and qualitative sample analysis

    See Also:
        - [`focoos.model_manager.ModelManager.get`][focoos.model_manager.ModelManager.get]: Model loading
        - [`focoos.trainer.evaluation`][focoos.trainer.evaluation]: Evaluation metrics
        - [`focoos.data.auto_dataset.AutoDataset`][focoos.data.auto_dataset.AutoDataset]: Dataset handling
        - [`focoos.ports.TrainerArgs`][focoos.ports.TrainerArgs]: Configuration parameters
    """
    logger.info(f"🚀 Starting validation: {model_name} with dataset: {dataset_name}")
    logger.info(f"📋 Configuration: {im_size}px, batch size {batch_size}")

    try:
        # Load model
        logger.info(f"🔄 Loading model: {model_name}")
        model = ModelManager.get(model_name)

        # Load dataset
        logger.info(f"🔄 Loading dataset: {dataset_name}")
        auto_dataset = AutoDataset(
            dataset_name=dataset_name,
            task=model.model_info.task,
            layout=dataset_layout,
            datasets_dir=datasets_dir or DATASETS_DIR,
        )

        # Get validation augmentations
        logger.info("🎨 Setting up validation augmentations")
        _, val_augs = get_default_by_task(task=model.model_info.task, resolution=im_size or model.model_info.im_size)
        valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

        # Configure validation arguments
        logger.info("⚙️ Configuring validation parameters")
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
        logger.info("🔍 Starting validation loop...")
        logger.info(f"📊 Validation samples: {len(valid_dataset)}")

        model.eval(trainer_args, valid_dataset)

        logger.info("✅ Validation completed successfully!")
        logger.info(f"📁 Output directory: {trainer_args.output_dir}")
        logger.info(f"🏷️ Run name: {trainer_args.run_name}")

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        raise
