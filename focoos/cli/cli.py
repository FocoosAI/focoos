"""Focoos Command Line Interface.

The Focoos CLI provides both Typer-style commands and a streamlined interface for computer vision tasks.

This module contains the main CLI application with commands for training, validation, prediction,
export, benchmarking, and system utilities.

Usage:
    ```bash
    focoos COMMAND [OPTIONS]
    ```

Available Commands:
    - `train`: Train a model on a dataset
    - `val`: Validate a model on a dataset
    - `predict`: Run inference on images
    - `export`: Export a model to different formats
    - `benchmark`: Benchmark model performance
    - `version`: Show Focoos version information
    - `checks`: Run system checks and display system information
    - `settings`: Show current Focoos configuration settings

Examples:
    Training a model:
    ```bash
    focoos train --model fai-detr-m-coco --dataset mydataset.zip --im-size 640
    ```

    Running inference:
    ```bash
    focoos predict --model fai-detr-m-coco --source image.jpg --im-size 640
    ```

    Validating a model:
    ```bash
    focoos val --model fai-detr-m-coco --dataset mydataset.zip --im-size 640
    ```

    Exporting a model:
    ```bash
    focoos export --model fai-detr-m-coco --format onnx --im-size 640
    ```

    Benchmarking performance:
    ```bash
    focoos benchmark --model fai-detr-m-coco --iterations 100 --device cuda
    ```

    System information:
    ```bash
    focoos checks
    focoos settings
    focoos version
    ```

Note:
    For command-specific help, use: `focoos COMMAND --help`

See Also:
    - [Training Guide](../training.md)
    - [Inference Guide](../inference.md)
"""

import uuid
from typing import Optional, cast, get_args

import typer
from typing_extensions import Annotated

from focoos.cli.commands import (
    benchmark_command,
    export_command,
    predict_command,
    train_command,
    val_command,
)
from focoos.cli.commands.hub import app as hub_app
from focoos.ports import (
    PREDICTIONS_DIR,
    DatasetLayout,
    DeviceType,
    ExportFormat,
    OptimizerType,
    RuntimeType,
    SchedulerType,
    get_gpus_count,
)
from focoos.utils.logger import get_logger

logger = get_logger("CLI")

app = typer.Typer(
    name="focoos",
    help=__doc__,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command("version")
def version():
    """Show Focoos version information.

    This command displays the current version of the Focoos library installed
    on the system. If the version cannot be determined, it shows "Unknown".

    Examples:
        ```bash
        focoos version
        ```

        Output:
        ```
        Focoos version: 0.15.0
        ```

    Raises:
        typer.Exit: If there's an error retrieving version information.

    See Also:
        - [`focoos checks`][focoos.cli.cli.checks]: For system diagnostics
        - [`focoos settings`][focoos.cli.cli.settings]: For configuration info
    """
    try:
        from focoos.utils.system import get_focoos_version

        version = get_focoos_version()
        typer.echo(f"Focoos version: {version}")
    except Exception:
        typer.echo("Focoos version: Unknown")


@app.command("checks")
def checks():
    """Run system checks and display system information.

    This command performs comprehensive system checks including hardware
    information, GPU availability, dependencies, and other system-level
    diagnostics relevant to Focoos operation.

    The system checks include:
        - Hardware information (CPU, memory, GPU)
        - CUDA availability and version
        - Python environment details
        - Focoos dependencies status
        - System compatibility verification

    Examples:
        ```bash
        focoos checks
        ```

        Output:
        ```
        Running system checks...

        System Information:
        ==================
        OS: Linux 5.4.0-74-generic
        Python: 3.8.10
        CUDA: 11.2.2
        GPU: NVIDIA GeForce RTX 3080
        Memory: 32GB
        ```

    Raises:
        typer.Exit: If system checks fail or encounter errors.

    See Also:
        - [`focoos version`][focoos.cli.cli.version]: For version info only
        - [`focoos settings`][focoos.cli.cli.settings]: For configuration details
    """
    typer.echo("Running system checks...")
    try:
        from focoos.utils.system import get_system_info

        system_info = get_system_info()
        system_info.pprint()
    except Exception as e:
        logger.error(f"Error running checks: {e}")
        raise typer.Exit(1)


@app.command("settings")
def settings():
    """Show current Focoos configuration settings.

    This command displays the current configuration settings for Focoos,
    including runtime type, host URL, API key status, log level, and
    other configuration parameters.

    Configuration includes:
        - **Runtime Type**: Backend engine for inference
        - **Host URL**: API endpoint for Focoos services
        - **API Key**: Authentication status (masked for security)
        - **Log Level**: Current logging verbosity
        - **Warmup Iterations**: Performance optimization setting

    Examples:
        ```bash
        focoos settings
        ```

        Output:
        ```
        Current Focoos settings:
        FOCOOS_RUNTIME_TYPE=ONNX_CUDA32
        FOCOOS_HOST_URL=https://api.focoos.ai
        FOCOOS_API_KEY=********************
        FOCOOS_LOG_LEVEL=INFO
        FOCOOS_WARMUP_ITER=3
        ```

    Raises:
        typer.Exit: If there's an error accessing configuration settings.

    See Also:
        - [`focoos checks`][focoos.cli.cli.checks]: For system diagnostics
    """
    try:
        from focoos.config import FOCOOS_CONFIG

        typer.echo("Current Focoos settings:")
        typer.echo(f"FOCOOS_RUNTIME_TYPE={FOCOOS_CONFIG.runtime_type}")
        typer.echo(f"FOCOOS_HOST_URL={FOCOOS_CONFIG.default_host_url}")
        typer.echo(f"FOCOOS_API_KEY={'*' * 20 if FOCOOS_CONFIG.focoos_api_key else 'Not set'}")
        typer.echo(f"FOCOOS_LOG_LEVEL={FOCOOS_CONFIG.focoos_log_level}")
        typer.echo(f"FOCOOS_WARMUP_ITER={FOCOOS_CONFIG.warmup_iter}")
    except Exception as e:
        logger.error(f"Error showing settings: {e}")
        raise typer.Exit(1)


@app.command("train")
def train(
    model: Annotated[str, typer.Option(help="Model name (required)")],
    dataset: Annotated[str, typer.Option(help="Dataset name (required)")],
    run_name: Annotated[Optional[str], typer.Option(help="Run name")] = None,
    datasets_dir: Annotated[
        Optional[str], typer.Option(help="Datasets directory (default: ~/FocoosAI/datasets/)")
    ] = None,
    dataset_layout: Annotated[DatasetLayout, typer.Option(help="Dataset layout")] = DatasetLayout.ROBOFLOW_COCO,
    im_size: Annotated[int, typer.Option(help="Image size")] = 640,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    ckpt_dir: Annotated[Optional[str], typer.Option(help="Checkpoint directory")] = None,
    init_checkpoint: Annotated[Optional[str], typer.Option(help="Initial checkpoint path")] = None,
    resume: Annotated[bool, typer.Option(help="Resume training from checkpoint")] = False,
    num_gpus: Annotated[int, typer.Option(help="Number of GPUs to use")] = get_gpus_count(),
    device: Annotated[str, typer.Option(help="Device to use")] = "cuda",
    workers: Annotated[int, typer.Option(help="Number of workers")] = 4,
    amp_enabled: Annotated[bool, typer.Option(help="Enable automatic mixed precision")] = True,
    ddp_broadcast_buffers: Annotated[bool, typer.Option(help="Broadcast buffers in DDP")] = False,
    ddp_find_unused: Annotated[bool, typer.Option(help="Find unused parameters in DDP")] = True,
    checkpointer_period: Annotated[int, typer.Option(help="Checkpoint save period")] = 1000,
    checkpointer_max_to_keep: Annotated[int, typer.Option(help="Maximum checkpoints to keep")] = 1,
    eval_period: Annotated[int, typer.Option(help="Evaluation period")] = 50,
    log_period: Annotated[int, typer.Option(help="Logging period")] = 20,
    samples: Annotated[int, typer.Option(help="Number of samples to log")] = 9,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    early_stop: Annotated[bool, typer.Option(help="Enable early stopping")] = True,
    patience: Annotated[int, typer.Option(help="Early stopping patience")] = 10,
    ema_enabled: Annotated[bool, typer.Option(help="Enable EMA")] = False,
    ema_decay: Annotated[float, typer.Option(help="EMA decay rate")] = 0.999,
    ema_warmup: Annotated[int, typer.Option(help="EMA warmup steps")] = 2000,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 5e-4,
    weight_decay: Annotated[float, typer.Option(help="Weight decay")] = 0.02,
    max_iters: Annotated[int, typer.Option(help="Maximum iterations")] = 3000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 16,
    scheduler: Annotated[str, typer.Option(help="Learning rate scheduler")] = "MULTISTEP",
    optimizer: Annotated[str, typer.Option(help="Optimizer type")] = "ADAMW",
    weight_decay_norm: Annotated[float, typer.Option(help="Weight decay for normalization layers")] = 0.0,
    weight_decay_embed: Annotated[float, typer.Option(help="Weight decay for embedding layers")] = 0.0,
    backbone_multiplier: Annotated[float, typer.Option(help="Learning rate multiplier for backbone")] = 0.1,
    decoder_multiplier: Annotated[float, typer.Option(help="Learning rate multiplier for decoder")] = 1.0,
    head_multiplier: Annotated[float, typer.Option(help="Learning rate multiplier for head")] = 1.0,
    freeze_bn: Annotated[bool, typer.Option(help="Freeze batch normalization layers")] = False,
    clip_gradients: Annotated[float, typer.Option(help="Gradient clipping value")] = 0.1,
    size_divisibility: Annotated[int, typer.Option(help="Size divisibility")] = 0,
    gather_metric_period: Annotated[int, typer.Option(help="Metric gathering period")] = 1,
    zero_grad_before_forward: Annotated[bool, typer.Option(help="Zero gradients before forward pass")] = False,
    sync_to_hub: Annotated[bool, typer.Option(help="Sync to Focoos Hub")] = False,
):
    """Train a model with comprehensive configuration options.

    This command initiates model training with extensive customization options
    for dataset handling, model architecture, optimization, and training dynamics.
    Supports distributed training, mixed precision, and various optimization strategies.

    Training Features:
        - **Multi-GPU Training**: Automatic distributed training support
        - **Mixed Precision**: Faster training with AMP (Automatic Mixed Precision)
        - **Early Stopping**: Prevent overfitting with validation-based stopping
        - **EMA (Exponential Moving Average)**: Improved model stability
        - **Flexible Scheduling**: Multiple learning rate schedules
        - **Checkpoint Management**: Automatic saving and resuming

    Args:
        model (str): Name of the model architecture to train (e.g., 'fai-detr-m-coco').
            Can be specified in several ways:
            - **Pretrained model**: Simple model name like 'fai-detr-m-coco'
            - **Hub model**: Format 'hub://<model-ref>' for models from Focoos Hub
            - **Default directory model**: Model name for models in default Focoos directory
        dataset (str): Name of the dataset to train on (e.g., 'mydataset.zip').
        run_name (Optional[str]): Optional name for the training run. If not provided,
            generates a unique name using model name and UUID.
        models_dir (Optional[str]): Directory to save models.
        datasets_dir (Optional[str]): Custom directory for datasets.
        dataset_layout (DatasetLayout): Layout format of the dataset. Defaults to ROBOFLOW_COCO.
        im_size (int): Input image size for training. Defaults to 640.
        output_dir (Optional[str]): Directory to save training outputs and logs.
        ckpt_dir (Optional[str]): Directory to save model checkpoints.
        init_checkpoint (Optional[str]): Path to initial checkpoint for transfer learning.
        resume (bool): Whether to resume training from the latest checkpoint. Defaults to False.
        num_gpus (int): Number of GPUs to use for training. Defaults to auto-detected count.
        device (str): Device type for training ('cuda' or 'cpu'). Defaults to 'cuda'.
        workers (int): Number of data loading workers. Defaults to 4.
        amp_enabled (bool): Enable Automatic Mixed Precision for faster training. Defaults to True.
        ddp_broadcast_buffers (bool): Whether to broadcast buffers in DistributedDataParallel.
            Defaults to False.
        ddp_find_unused (bool): Whether to find unused parameters in DDP. Defaults to True.
        checkpointer_period (int): Frequency of checkpoint saving (in iterations). Defaults to 1000.
        checkpointer_max_to_keep (int): Maximum number of checkpoints to retain. Defaults to 1.
        eval_period (int): Frequency of model evaluation (in iterations). Defaults to 50.
        log_period (int): Frequency of logging metrics (in iterations). Defaults to 20.
        samples (int): Number of sample images to log during training. Defaults to 9.
        seed (int): Random seed for reproducible training. Defaults to 42.
        early_stop (bool): Enable early stopping based on validation metrics. Defaults to True.
        patience (int): Number of evaluations to wait before early stopping. Defaults to 10.
        ema_enabled (bool): Enable Exponential Moving Average of model weights. Defaults to False.
        ema_decay (float): Decay rate for EMA. Defaults to 0.999.
        ema_warmup (int): Number of warmup steps for EMA. Defaults to 2000.
        learning_rate (float): Initial learning rate for optimization. Defaults to 5e-4.
        weight_decay (float): L2 regularization weight decay. Defaults to 0.02.
        max_iters (int): Maximum number of training iterations. Defaults to 3000.
        batch_size (int): Training batch size. Defaults to 16.
        scheduler (str): Learning rate scheduler type ('MULTISTEP', 'COSINE', etc.).
            Defaults to 'MULTISTEP'.
        optimizer (str): Optimizer type ('ADAMW', 'SGD', etc.). Defaults to 'ADAMW'.
        weight_decay_norm (float): Weight decay for normalization layers. Defaults to 0.0.
        weight_decay_embed (float): Weight decay for embedding layers. Defaults to 0.0.
        backbone_multiplier (float): Learning rate multiplier for backbone layers. Defaults to 0.1.
        decoder_multiplier (float): Learning rate multiplier for decoder layers. Defaults to 1.0.
        head_multiplier (float): Learning rate multiplier for head layers. Defaults to 1.0.
        freeze_bn (bool): Whether to freeze batch normalization layers. Defaults to False.
        clip_gradients (float): Gradient clipping threshold. Defaults to 0.1.
        size_divisibility (int): Image size divisibility constraint. Defaults to 0.
        gather_metric_period (int): Frequency of metric gathering (in iterations). Defaults to 1.
        zero_grad_before_forward (bool): Whether to zero gradients before forward pass.
            Defaults to False.
        sync_to_hub (bool): Whether to sync model to Focoos Hub. Defaults to False.

    Examples:
        Basic training with pretrained model:
        ```bash
        focoos train --model fai-detr-m-coco --dataset mydataset.zip
        ```

        Training with model from Focoos Hub:
        ```bash
        focoos train --model hub://<model-ref> --dataset mydataset.zip
        ```

        Training with local model in default Focoos directory:
        ```bash
        focoos train --model my-saved-model --dataset mydataset.zip
        ```

        Advanced training with custom parameters:
        ```bash
        focoos train --model fai-detr-m-coco --dataset mydataset.zip \
                     --im-size 800 --batch-size 32 --learning-rate 1e-4 \
                     --max-iters 5000 --early-stop --patience 20
        ```

        Multi-GPU training with mixed precision:
        ```bash
        focoos train --model fai-detr-m-coco --dataset custom_dataset \
                     --num-gpus 4 --amp-enabled --batch-size 64
        ```

        Resume training from checkpoint:
        ```bash
        focoos train --model fai-detr-m-coco --dataset mydataset.zip \
                     --resume --ckpt-dir ./checkpoints
        ```

    Raises:
        AssertionError: If device, scheduler, or optimizer parameters are invalid.
        FileNotFoundError: If dataset or checkpoint files are not found.
        RuntimeError: If GPU resources are insufficient or CUDA is unavailable.

    See Also:
        - [`focoos val`][focoos.cli.cli.val]: For model validation
        - [`focoos export`][focoos.cli.cli.export]: For model export
        - [Training Guide](../training.md): For detailed training documentation
    """
    typer.echo("🔍 Training arguments:")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  Dataset: {dataset}")
    typer.echo(f"  Dataset layout: {dataset_layout}")
    typer.echo(f"  Image size: {im_size}")
    typer.echo(f"  Run name: {run_name}")
    typer.echo(f"  Output dir: {output_dir}")
    typer.echo(f"  Checkpoint dir: {ckpt_dir}")
    typer.echo(f"  Init checkpoint: {init_checkpoint}")
    typer.echo(f"  Resume: {resume}")
    typer.echo(f"  Num GPUs: {num_gpus}")
    typer.echo(f"  Device: {device}")
    typer.echo(f"  Workers: {workers}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Scheduler: {scheduler}")
    typer.echo(f"  Optimizer: {optimizer}")
    typer.echo(f"  Weight decay norm: {weight_decay_norm}")
    typer.echo(f"  Weight decay embed: {weight_decay_embed}")
    typer.echo(f"  Backbone multiplier: {backbone_multiplier}")
    typer.echo(f"  Decoder multiplier: {decoder_multiplier}")
    typer.echo(f"  Head multiplier: {head_multiplier}")

    try:
        # Cast to proper literal types
        validated_device = cast(DeviceType, device)
        assert device in get_args(DeviceType)
        validated_scheduler = cast(SchedulerType, scheduler.upper())
        assert scheduler in get_args(SchedulerType)
        validated_optimizer = cast(OptimizerType, optimizer.upper())
        assert optimizer in get_args(OptimizerType)

        train_command(
            model_name=model,
            dataset_name=dataset,
            dataset_layout=dataset_layout,
            im_size=im_size,
            run_name=run_name or f"{model}-{uuid.uuid4()}",
            output_dir=output_dir,
            ckpt_dir=ckpt_dir,
            init_checkpoint=init_checkpoint,
            resume=resume,
            num_gpus=num_gpus,
            device=validated_device,
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
            batch_size=batch_size,
            scheduler=validated_scheduler,
            optimizer=validated_optimizer,
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
            datasets_dir=datasets_dir,
        )
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}")


@app.command("val")
def val(
    model: Annotated[str, typer.Option(help="Model name (required)")],
    dataset: Annotated[str, typer.Option(help="Dataset name (required)")],
    datasets_dir: Annotated[
        Optional[str], typer.Option(help="Datasets directory (default: ~/FocoosAI/datasets/)")
    ] = None,
    run_name: Annotated[Optional[str], typer.Option(help="Run name")] = None,
    dataset_layout: Annotated[DatasetLayout, typer.Option(help="Dataset layout")] = DatasetLayout.ROBOFLOW_COCO,
    im_size: Annotated[int, typer.Option(help="Image size")] = 640,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    ckpt_dir: Annotated[Optional[str], typer.Option(help="Checkpoint directory")] = None,
    init_checkpoint: Annotated[Optional[str], typer.Option(help="Initial checkpoint")] = None,
    resume: Annotated[bool, typer.Option(help="Resume training")] = False,
    num_gpus: Annotated[int, typer.Option(help="Number of GPUs")] = get_gpus_count(),
    device: Annotated[str, typer.Option(help="Device")] = "cuda",
    workers: Annotated[int, typer.Option(help="Number of workers")] = 4,
    amp_enabled: Annotated[bool, typer.Option(help="Enable AMP")] = True,
    ddp_broadcast_buffers: Annotated[bool, typer.Option(help="DDP broadcast buffers")] = False,
    ddp_find_unused: Annotated[bool, typer.Option(help="DDP find unused")] = True,
    checkpointer_period: Annotated[int, typer.Option(help="Checkpointer period")] = 1000,
    checkpointer_max_to_keep: Annotated[int, typer.Option(help="Checkpointer max to keep")] = 1,
    eval_period: Annotated[int, typer.Option(help="Evaluation period")] = 50,
    log_period: Annotated[int, typer.Option(help="Log period")] = 20,
    samples: Annotated[int, typer.Option(help="Number of samples")] = 9,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    early_stop: Annotated[bool, typer.Option(help="Enable early stopping")] = True,
    patience: Annotated[int, typer.Option(help="Early stopping patience")] = 10,
    ema_enabled: Annotated[bool, typer.Option(help="Enable EMA")] = False,
    ema_decay: Annotated[float, typer.Option(help="EMA decay")] = 0.999,
    ema_warmup: Annotated[int, typer.Option(help="EMA warmup")] = 2000,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 5e-4,
    weight_decay: Annotated[float, typer.Option(help="Weight decay")] = 0.02,
    max_iters: Annotated[int, typer.Option(help="Maximum iterations")] = 3000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 16,
    scheduler: Annotated[str, typer.Option(help="Scheduler type")] = "MULTISTEP",
    optimizer: Annotated[str, typer.Option(help="Optimizer type")] = "ADAMW",
    weight_decay_norm: Annotated[float, typer.Option(help="Weight decay for normalization layers")] = 0.0,
    weight_decay_embed: Annotated[float, typer.Option(help="Weight decay for embedding layers")] = 0.0,
    backbone_multiplier: Annotated[float, typer.Option(help="Backbone learning rate multiplier")] = 0.1,
    decoder_multiplier: Annotated[float, typer.Option(help="Decoder learning rate multiplier")] = 1.0,
    head_multiplier: Annotated[float, typer.Option(help="Head learning rate multiplier")] = 1.0,
    freeze_bn: Annotated[bool, typer.Option(help="Freeze batch normalization")] = False,
    clip_gradients: Annotated[float, typer.Option(help="Gradient clipping value")] = 0.1,
    size_divisibility: Annotated[int, typer.Option(help="Size divisibility")] = 0,
    gather_metric_period: Annotated[int, typer.Option(help="Gather metric period")] = 1,
    zero_grad_before_forward: Annotated[bool, typer.Option(help="Zero gradients before forward pass")] = False,
):
    """Validate a model on a dataset with comprehensive evaluation metrics.

    This command performs model validation/evaluation on a specified dataset,
    computing various metrics such as mAP, precision, recall, and other
    task-specific evaluation measures. Supports the same configuration options
    as training for consistency in evaluation setup.

    Validation Metrics:
        - **mAP (mean Average Precision)**: Overall detection accuracy
        - **Precision/Recall**: Per-class and overall performance
        - **F1-Score**: Harmonic mean of precision and recall
        - **IoU (Intersection over Union)**: Bounding box accuracy
        - **Inference Speed**: FPS and latency measurements

    Args:
        model (str): Name of the model to validate (e.g., 'fai-detr-m-coco').
            Can be specified in several ways:
            - **Pretrained model**: Simple model name like 'fai-detr-m-coco'
            - **Hub model**: Format 'hub://<model-ref>' for models from Focoos Hub
            - **Default directory model**: Model name for models in default Focoos directory
        dataset (str): Name of the dataset for validation (e.g., 'mydataset.zip').
        run_name (Optional[str]): Optional name for the validation run. If not provided,
            generates a unique name using model name and UUID.
        dataset_layout (DatasetLayout): Layout format of the dataset. Defaults to ROBOFLOW_COCO.
        im_size (int): Input image size for validation. Defaults to 640.
        output_dir (Optional[str]): Directory to save validation outputs and results.
        ckpt_dir (Optional[str]): Directory containing model checkpoints.
        init_checkpoint (Optional[str]): Path to specific checkpoint for validation.
        resume (bool): Whether to resume from checkpoint (typically not used in validation).
        num_gpus (int): Number of GPUs to use for validation. Defaults to auto-detected count.
        device (str): Device type for validation ('cuda' or 'cpu'). Defaults to 'cuda'.
        workers (int): Number of data loading workers. Defaults to 4.
        amp_enabled (bool): Enable Automatic Mixed Precision for faster inference. Defaults to True.
        ddp_broadcast_buffers (bool): Whether to broadcast buffers in DistributedDataParallel.
            Defaults to False.
        ddp_find_unused (bool): Whether to find unused parameters in DDP. Defaults to True.
        checkpointer_period (int): Checkpoint saving frequency (not typically used in validation).
        checkpointer_max_to_keep (int): Maximum checkpoints to keep.
        eval_period (int): Frequency of evaluation logging (in iterations). Defaults to 50.
        log_period (int): Frequency of metric logging (in iterations). Defaults to 20.
        samples (int): Number of sample images to visualize during validation. Defaults to 9.
        seed (int): Random seed for reproducible validation. Defaults to 42.
        early_stop (bool): Enable early stopping (not typically used in validation).
        patience (int): Early stopping patience.
        ema_enabled (bool): Use Exponential Moving Average weights if available.
        ema_decay (float): EMA decay rate.
        ema_warmup (int): EMA warmup steps.
        learning_rate (float): Learning rate (not used in validation).
        weight_decay (float): Weight decay (not used in validation).
        max_iters (int): Maximum validation iterations.
        batch_size (int): Validation batch size. Defaults to 16.
        scheduler (str): Scheduler type (not used in validation).
        optimizer (str): Optimizer type (not used in validation).
        weight_decay_norm (float): Weight decay for normalization layers.
        weight_decay_embed (float): Weight decay for embedding layers.
        backbone_multiplier (float): Backbone learning rate multiplier.
        decoder_multiplier (float): Decoder learning rate multiplier.
        head_multiplier (float): Head learning rate multiplier.
        freeze_bn (bool): Whether to freeze batch normalization layers.
        clip_gradients (float): Gradient clipping threshold.
        size_divisibility (int): Image size divisibility constraint.
        gather_metric_period (int): Frequency of metric gathering.
        zero_grad_before_forward (bool): Whether to zero gradients before forward pass.
        datasets_dir (Optional[str]): Custom directory for datasets.

    Examples:
        Basic validation with pretrained model:
        ```bash
        focoos val --model fai-detr-m-coco --dataset mydataset.zip
        ```

        Validation with model from Focoos Hub:
        ```bash
        focoos val --model hub://<model-ref> --dataset mydataset.zip
        ```

        Validation with local model in default Focoos directory:
        ```bash
        focoos val --model my-checkpoint-model --dataset mydataset.zip
        ```

        Validation with specific checkpoint:
        ```bash
        focoos val --model fai-detr-m-coco --dataset mydataset.zip \
                   --init-checkpoint path/to/checkpoint.pth --im-size 800
        ```

        Multi-GPU validation:
        ```bash
        focoos val --model fai-detr-m-coco --dataset large_dataset \
                   --num-gpus 2 --batch-size 32
        ```

        Validation with custom output directory:
        ```bash
        focoos val --model fai-detr-m-coco --dataset mydataset.zip \
                   --output-dir ./validation_results
        ```

    Raises:
        AssertionError: If device, scheduler, or optimizer parameters are invalid.
        FileNotFoundError: If dataset or checkpoint files are not found.
        RuntimeError: If GPU resources are insufficient or model loading fails.

    See Also:
        - [`focoos train`][focoos.cli.cli.train]: For model training
        - [`focoos predict`][focoos.cli.cli.predict]: For inference
        - [Validation Guide](../validation.md): For detailed validation documentation
    """

    typer.echo("🔍 Validation arguments:")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  Dataset: {dataset}")
    typer.echo(f"  Dataset layout: {dataset_layout}")
    typer.echo(f"  Image size: {im_size}")
    typer.echo(f"  Run name: {run_name}")
    typer.echo(f"  Output dir: {output_dir}")
    typer.echo(f"  Checkpoint dir: {ckpt_dir}")
    typer.echo(f"  Init checkpoint: {init_checkpoint}")
    typer.echo(f"  Resume: {resume}")
    typer.echo(f"  Num GPUs: {num_gpus}")
    typer.echo(f"  Device: {device}")
    typer.echo(f"  Workers: {workers}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Learning rate: {learning_rate}")
    typer.echo(f"  Weight decay: {weight_decay}")
    typer.echo(f"  Max iterations: {max_iters}")
    typer.echo(f"  Scheduler: {scheduler}")
    typer.echo(f"  Optimizer: {optimizer}")
    typer.echo(f"  Weight decay norm: {weight_decay_norm}")
    typer.echo(f"  Weight decay embed: {weight_decay_embed}")
    typer.echo(f"  Backbone multiplier: {backbone_multiplier}")
    typer.echo(f"  Decoder multiplier: {decoder_multiplier}")
    typer.echo(f"  Head multiplier: {head_multiplier}")
    typer.echo(f"  Freeze BN: {freeze_bn}")

    try:
        validated_device = cast(DeviceType, device)
        assert device in get_args(DeviceType)
        validated_scheduler = cast(SchedulerType, scheduler.upper())
        assert scheduler in get_args(SchedulerType)
        validated_optimizer = cast(OptimizerType, optimizer.upper())
        assert optimizer in get_args(OptimizerType)

        val_command(
            model_name=model,
            dataset_name=dataset,
            dataset_layout=dataset_layout,
            im_size=im_size,
            run_name=run_name or f"{model}-{uuid.uuid4()}",
            output_dir=output_dir,
            ckpt_dir=ckpt_dir,
            init_checkpoint=init_checkpoint,
            resume=resume,
            num_gpus=num_gpus,
            device=validated_device,
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
            batch_size=batch_size,
            scheduler=validated_scheduler,
            optimizer=validated_optimizer,
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
            datasets_dir=datasets_dir,
        )
    except Exception as e:
        typer.echo(f"❌ Validation failed: {e}")


@app.command("predict")
def predict(
    source: Annotated[str, typer.Option(help="Image/video path or URL (required)")],
    model: Annotated[str, typer.Option(help="Model name or path (required)")] = "fai-detr-l-obj365",
    runtime: Annotated[
        Optional[RuntimeType], typer.Option(help="Runtime type (If not provided, torch will be used)")
    ] = None,
    im_size: Annotated[Optional[int], typer.Option(help="Image size")] = 640,
    conf: Annotated[Optional[float], typer.Option(help="Confidence threshold")] = 0.5,
    output_dir: Annotated[
        Optional[str], typer.Option(help="Directory to save results (default: ~/FocoosAI/predictions/)")
    ] = PREDICTIONS_DIR,
    save: Annotated[Optional[bool], typer.Option(help="Save annotated image results")] = True,
    save_json: Annotated[Optional[bool], typer.Option(help="Save detections as JSON")] = True,
    save_masks: Annotated[Optional[bool], typer.Option(help="Save masks as separate images")] = True,
):
    """Run inference on images with flexible output options.

    This command performs inference (prediction) on input images using a trained model.
    Supports various output formats including annotated images, JSON results, and mask files.
    Results are always printed to the console.

    Supported Input:
        - **Single Images**: Local image files (loaded via `image_loader()`)
        - **URLs**: Remote image files accessible via URL

    Output Options:
        - **Console Output**: Detection results printed to terminal (always enabled)
        - **Annotated Images**: Visual results with bounding boxes and labels (if `save=True`)
        - **JSON Files**: Machine-readable detection results with coordinates and confidence (if `save_json=True`)
        - **Mask Images**: Individual mask files as PNG images (if `save_masks=True` and model produces masks)

    Args:
        model (str): Name of the model or path to model file for inference.
            Can be specified in several ways:
            - **Pretrained model**: Simple model name like 'fai-detr-m-coco'
            - **Hub model**: Format 'hub://<model-ref>' for models from Focoos Hub
            - **Default directory model**: Model name for models in default Focoos directory
        source (str): Path to input image file or URL. Must be a single image file.
        models_dir (Optional[str]): Directory containing model files.
        runtime (RuntimeType): Runtime backend for inference. Defaults to ONNX_CUDA32.
            Options include ONNX_CUDA32, ONNX_CPU, PYTORCH, etc.
        im_size (Optional[int]): Input image size for inference. Defaults to 640.
            Images will be resized to this size while maintaining aspect ratio.
        conf (Optional[float]): Confidence threshold for detections. Defaults to 0.25.
            Only detections above this threshold will be reported.
        save (Optional[bool]): Whether to save annotated images with detection overlays.
            Defaults to True.
        output_dir (Optional[str]): Directory to save all inference results.
        save_json (Optional[bool]): Whether to save detection results in JSON format.
            Defaults to True.
        save_masks (Optional[bool]): Whether to save segmentation masks as separate PNG images.
            Defaults to True. Only applies if the model produces masks.

    Examples:
        Basic image inference with pretrained model:
        ```bash
        focoos predict --model fai-detr-m-coco --source image.jpg
        ```

        Inference with model from Focoos Hub:
        ```bash
        focoos predict --model hub://<model-ref> --source image.jpg
        ```

        Inference with local model in default Focoos directory:
        ```bash
        focoos predict --model my-trained-model --source image.jpg
        ```

        Image inference with custom confidence:
        ```bash
        focoos predict --model fai-detr-m-coco --source image.jpg --conf 0.5
        ```

        Inference with custom output directory:
        ```bash
        focoos predict --model fai-detr-m-coco --source image.jpg \
                       --output-dir ./my_results --runtime ONNX_CPU
        ```

        URL inference:
        ```bash
        focoos predict --model fai-detr-m-coco \
                       --source https://example.com/image.jpg
        ```

        Only console output (no file saving):
        ```bash
        focoos predict --model fai-detr-m-coco --source image.jpg \
                       --save false --save-json false --save-masks false
        ```

    Output:
        The command prints detection results to the console and optionally saves files:

        Console output example:
        ```
        ==================================================
        DETECTION RESULTS
        ==================================================
        Found 2 detections:

          1. person
             Confidence: 0.856
             Bbox: [245, 120, 380, 450]
             Size: 135 x 330

          2. car
             Confidence: 0.742
             Bbox: [50, 200, 200, 300]
             Size: 150 x 100
        ==================================================
        ```

        File outputs (if enabled):
        - `{source_name}_annotated.{ext}`: Annotated image with bounding boxes
        - `{source_name}_detections.json`: Structured detection data
        - `{source_name}_masks/mask_N.png`: Individual mask files (if applicable)

    Note:
        The function currently processes single images only. For batch processing,
        the command needs to be run multiple times. All detection results are
        always printed to the console regardless of save options.

    See Also:
        - [`focoos val`][focoos.cli.cli.val]: For model validation
        - [`focoos benchmark`][focoos.cli.cli.benchmark]: For performance testing
        - [Inference Guide](../inference.md): For detailed inference documentation
    """
    typer.echo(
        f"🔮 Starting predict - Model: {model}, Source: {source}, Runtime: {runtime}, Image size: {im_size}, Conf: {conf}, Save: {save}, Output dir: {output_dir}, Save json: {save_json}, Save masks: {save_masks}"
    )
    try:
        predict_command(
            model_name=model,
            source=source,
            runtime=runtime,
            im_size=im_size,
            conf=conf,
            save=save,
            output_dir=output_dir,
            save_json=save_json,
            save_masks=save_masks,
        )
    except Exception as e:
        typer.echo(f"❌ Predict failed: {e}")


@app.command("export")
def export(
    model: Annotated[str, typer.Option(help="Model name (required)")],
    format: Annotated[Optional[ExportFormat], typer.Option(help="Export format")] = ExportFormat.ONNX,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    device: Annotated[Optional[str], typer.Option(help="Device (cuda or cpu)")] = "cuda",
    onnx_opset: Annotated[Optional[int], typer.Option(help="ONNX opset version")] = 17,
    im_size: Annotated[Optional[int], typer.Option(help="Image size for export")] = 640,
    overwrite: Annotated[Optional[bool], typer.Option(help="Overwrite existing files")] = False,
):
    """Export a trained model to various deployment formats.

    This command exports a Focoos model to different formats suitable for
    deployment in various environments. Currently supports ONNX and TorchScript
    formats for flexible deployment across different platforms and frameworks.

    Supported Export Formats:
        - **ONNX**: Cross-platform neural network format for interoperability
        - **TorchScript**: PyTorch's native serialization format for production

    Platform Compatibility:
        | Format | Linux | Windows | macOS | Mobile | Edge | Description |
        |--------|-------|---------|-------|--------|------|-------------|
        | ONNX   | ✅    | ✅      | ✅    | ✅     | ✅   | Universal format |
        | TorchScript | ✅ | ✅     | ✅    | ✅     | ✅   | PyTorch native |

    Args:
        model (str): Name of the Focoos model to export (e.g., 'fai-detr-m-coco').
            Can be specified in several ways:
            - **Pretrained model**: Simple model name like 'fai-detr-m-coco'
            - **Hub model**: Format 'hub://<model-ref>' for models from Focoos Hub
            - **Local model**: Model name with --models-dir pointing to custom directory
            - **Default directory model**: Model name for models in default Focoos directory
        models_dir (Optional[str]): Directory containing model files.
        format (Optional[ExportFormat]): Target export format. Defaults to ONNX.
            Available formats: 'onnx', 'torchscript'.
        output_dir (Optional[str]): Directory to save the exported model files.
            If not specified, uses a default export directory.
        device (Optional[str]): Device to use for export process ('cuda' or 'cpu').
            Defaults to 'cuda'.
        onnx_opset (Optional[int]): ONNX opset version for ONNX exports. Defaults to 17.
            Higher versions support more operations but may have compatibility issues.
            Only applies to ONNX format exports.
        im_size (Optional[int]): Input image size for the exported model. Defaults to 640.
            This determines the input tensor shape for the exported model.
        overwrite (Optional[bool]): Whether to overwrite existing exported files.
            Defaults to False. If False, export will fail if output files already exist.

    Examples:
        Basic ONNX export with pretrained model:
        ```bash
        focoos export --model fai-detr-m-coco
        ```

        Export model from Focoos Hub:
        ```bash
        focoos export --model hub://<model-ref> --format onnx
        ```

        Export local model from custom directory:
        ```bash
        focoos export --model my-fine-tuned-model --models-dir ./trained_models
        ```

        Export local model from default Focoos directory:
        ```bash
        focoos export --model my-checkpoint-model --format torchscript
        ```

        Export to TorchScript:
        ```bash
        focoos export --model fai-detr-m-coco --format torchscript
        ```

        Export with custom image size:
        ```bash
        focoos export --model fai-detr-m-coco --format onnx --im-size 800
        ```

        Export to custom directory with overwrite:
        ```bash
        focoos export --model fai-detr-m-coco --format onnx \
                      --output-dir ./exported_models --overwrite
        ```

        CPU-based export:
        ```bash
        focoos export --model fai-detr-m-coco --format torchscript \
                      --device cpu --im-size 416
        ```

    Raises:
        AssertionError: If device parameter is not valid.
        ValueError: If export format is not supported.
        RuntimeError: If export process fails due to model incompatibility.
        FileExistsError: If output files exist and overwrite is disabled.

    Note:
        ONNX format provides broader compatibility across different frameworks and
        deployment environments. TorchScript format is optimized for PyTorch-based
        deployment scenarios and may offer better performance in PyTorch environments.

    See Also:
        - [`focoos predict`][focoos.cli.cli.predict]: For using exported models
        - [`focoos benchmark`][focoos.cli.cli.benchmark]: For testing exported models
    """
    typer.echo(
        f"📦 Starting export - Model: {model}, Format: {format}, Output dir: {output_dir}, Device: {device}, ONNX opset: {onnx_opset}, Image size: {im_size}, Overwrite: {overwrite}"
    )
    try:
        validated_device = cast(DeviceType, device)
        assert device in get_args(DeviceType)
        export_command(
            model_name=model,
            format=format,
            output_dir=output_dir,
            device=validated_device,
            onnx_opset=onnx_opset,
            im_size=im_size,
            overwrite=overwrite,
        )
    except Exception as e:
        typer.echo(f"❌ Export failed: {e}")


@app.command("benchmark")
def benchmark(
    model: Annotated[str, typer.Option(help="Model name or path (required)")],
    im_size: Annotated[Optional[int], typer.Option(help="Image size for benchmarking")] = None,
    iterations: Annotated[Optional[int], typer.Option(help="Number of benchmark iterations")] = None,
    device: Annotated[str, typer.Option(help="Device for benchmarking (cuda or cpu)")] = "cuda",
):
    """Benchmark model performance with detailed metrics.

    This command performs comprehensive performance benchmarking of a model,
    measuring inference speed, memory usage, and throughput across multiple
    iterations. Provides detailed statistics including mean, median, and
    percentile measurements for performance analysis.

    Benchmark Metrics:
        - **Inference Time**: Per-image processing time (ms)
        - **Throughput**: Frames per second (FPS)
        - **Memory Usage**: Peak GPU/CPU memory consumption
        - **Latency Statistics**: P50, P95, P99 percentiles
        - **Model Loading Time**: Initialization overhead
        - **Warmup Performance**: Cold vs. warm inference speeds

    Performance Factors:
        - **Image Size**: Larger images → slower inference
        - **Model Complexity**: More parameters → higher latency
        - **Device Type**: GPU vs. CPU performance differences
        - **Batch Size**: Single vs. batch inference comparison
        - **Runtime Backend**: ONNX vs. PyTorch performance

    Args:
        model (str): Name of the model or path to model file for benchmarking.
            Can be specified in several ways:
            - **Pretrained model**: Simple model name like 'fai-detr-m-coco'
            - **Hub model**: Format 'hub://<model-ref>' for models from Focoos Hub
            - **Local model**: Model name with --models-dir pointing to custom directory
            - **Default directory model**: Model name for models in default Focoos directory
        models_dir (Optional[str]): Directory containing model files.
        im_size (Optional[int]): Input image size for benchmarking. If not specified,
            uses the model's default input size. Larger sizes typically
            result in slower inference but may improve accuracy.
        iterations (Optional[int]): Number of benchmark iterations to run. If not specified,
            uses a default number of iterations suitable for reliable statistics.
            More iterations provide more accurate timing measurements.
        device (str): Device to use for benchmarking ('cuda' or 'cpu'). Defaults to 'cuda'.
            GPU benchmarking typically shows better performance but requires CUDA availability.

    Examples:
        Basic benchmarking with pretrained model:
        ```bash
        focoos benchmark --model fai-detr-m-coco
        ```

        Benchmark model from Focoos Hub:
        ```bash
        focoos benchmark --model hub://<model-ref> --iterations 100
        ```

        Benchmark local model from custom directory:
        ```bash
        focoos benchmark --model my-model --models-dir ./custom_models --im-size 640
        ```

        Benchmark local model from default Focoos directory:
        ```bash
        focoos benchmark --model my-exported-model --iterations 200
        ```

        Benchmark with specific parameters:
        ```bash
        focoos benchmark --model fai-detr-m-coco --im-size 800 --iterations 100
        ```

        CPU benchmarking:
        ```bash
        focoos benchmark --model fai-detr-m-coco --device cpu --iterations 50
        ```

        Comprehensive benchmark suite:
        ```bash
        # Test different image sizes
        focoos benchmark --model fai-detr-m-coco --im-size 416
        focoos benchmark --model fai-detr-m-coco --im-size 640
        focoos benchmark --model fai-detr-m-coco --im-size 800
        ```

    Output:
        The command prints benchmark results to the console:

        ```
        Benchmark Results:
        ==================
        Model: fai-detr-m-coco
        Device: CUDA (GeForce RTX 3080)
        Image Size: 640x640
        Iterations: 100

        Performance Metrics:
        - Average Inference Time: 12.5ms
        - Throughput (FPS): 80.0
        - P50 Latency: 11.8ms
        - P95 Latency: 15.2ms
        - P99 Latency: 18.1ms
        - Peak GPU Memory: 2.1GB
        - Model Loading Time: 1.2s
        ```

    Raises:
        AssertionError: If device parameter is not valid.
        FileNotFoundError: If model file is not found.
        RuntimeError: If device is unavailable or out of memory.
        ImportError: If required runtime dependencies are missing.

    Note:
        Benchmark results can vary based on system load, thermal throttling,
        and hardware specifications. For consistent results, ensure the system
        is not under heavy load during benchmarking. Results are automatically
        saved to a benchmark report file for later analysis.

    See Also:
        - [`focoos predict`][focoos.cli.cli.predict]: For actual inference
        - [`focoos export`][focoos.cli.cli.export]: For optimized model formats
    """
    typer.echo(f"📦 Starting benchmark - Model: {model}, Iterations: {iterations}, Size: {im_size}, Device: {device}")
    try:
        validated_device = cast(DeviceType, device)
        assert device in get_args(DeviceType)
        benchmark_command(
            model_name=model,
            iterations=iterations,
            im_size=im_size,
            device=validated_device,
        )
    except Exception as e:
        typer.echo(f"❌ Benchmark failed: {e}")


app.add_typer(hub_app, name="hub")
