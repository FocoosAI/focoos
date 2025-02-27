import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

S3_URL_REGEX = re.compile(r"^s3://" r"(?P<bucket>[a-zA-Z0-9.-]+)/" r"(?P<path>.+(\.tar\.gz|\.zip)?)$")

DEV_API_URL = "https://api.dev.focoos.ai/v0"
PROD_API_URL = "https://api.focoos.ai/v0"
LOCAL_API_URL = "http://localhost:8501/v0"


class FocoosBaseModel(BaseModel):
    @classmethod
    def from_json(cls, data: Union[str, dict]):
        if isinstance(data, str):
            with open(data, encoding="utf-8") as f:
                data_dict = json.load(f)
        else:
            data_dict = data
        return cls.model_validate(data_dict)


class ModelStatus(str, Enum):
    """Status of a Focoos model during its lifecycle.

    Values:
        - CREATED: Model has been created
        - TRAINING_STARTING: Training is about to start
        - TRAINING_RUNNING: Training is in progress
        - TRAINING_ERROR: Training encountered an error
        - TRAINING_COMPLETED: Training finished successfully
        - TRAINING_STOPPED: Training was stopped
        - DEPLOYED: Model is deployed
        - DEPLOY_ERROR: Deployment encountered an error

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")
        model = focoos.get_remote_model("my-model")

        if model.status == ModelStatus.DEPLOYED:
            print("Model is deployed and ready for inference")
        elif model.status == ModelStatus.TRAINING_RUNNING:
            print("Model is currently training")
        elif model.status == ModelStatus.TRAINING_ERROR:
            print("Model training encountered an error")
        ```
    """

    CREATED = "CREATED"
    TRAINING_STARTING = "TRAINING_STARTING"
    TRAINING_RUNNING = "TRAINING_RUNNING"
    TRAINING_ERROR = "TRAINING_ERROR"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    TRAINING_STOPPED = "TRAINING_STOPPED"
    DEPLOYED = "DEPLOYED"
    DEPLOY_ERROR = "DEPLOY_ERROR"


class DatasetLayout(str, Enum):
    """Supported dataset formats in Focoos.

    Values:
        - ROBOFLOW_COCO: Roboflow COCO format
        - ROBOFLOW_SEG: Roboflow segmentation format
        - CATALOG: Catalog format
        - SUPERVISELY: Supervisely format

    Example:
        ```python
        from focoos import Focoos
        from focoos.ports import DatasetLayout

        focoos = Focoos(api_key="<YOUR-API-KEY>")
        datasets = focoos.list_shared_datasets()

        for dataset in datasets:
            if dataset.layout == DatasetLayout.ROBOFLOW_COCO:
                print(f"Dataset {dataset.name} uses ROBOFLOW_COCO format")
        ```
    """

    ROBOFLOW_COCO = "roboflow_coco"
    ROBOFLOW_SEG = "roboflow_seg"
    CATALOG = "catalog"
    SUPERVISELY = "supervisely"


class FocoosTask(str, Enum):
    """Types of computer vision tasks supported by Focoos.

    Values:
        - DETECTION: Object detection
        - SEMSEG: Semantic segmentation
        - INSTANCE_SEGMENTATION: Instance segmentation

    Example:
        ```python
        from focoos import Focoos, FocoosTask

        # Initialize Focoos client
        focoos = Focoos(api_key="<YOUR-API-KEY>")

        # Get list of models
        models = focoos.list_models()

        # Print task type for each model
        for model in models:
            if model.task == FocoosTask.DETECTION:
                print(f"{model.name} is a detection model")
            elif model.task == FocoosTask.SEMSEG:
                print(f"{model.name} is a semantic segmentation model")
            elif model.task == FocoosTask.INSTANCE_SEGMENTATION:
                print(f"{model.name} is an instance segmentation model")
        ```
    """

    DETECTION = "detection"
    SEMSEG = "semseg"
    INSTANCE_SEGMENTATION = "instseg"


class Hyperparameters(FocoosBaseModel):
    """Model training hyperparameters configuration.

    Attributes:
        batch_size (int): Number of images processed in each training iteration. Range: 1-32.
            Larger batch sizes require more GPU memory but can speed up training.

        eval_period (int): Number of iterations between model evaluations. Range: 50-2000.
            Controls how frequently validation is performed during training.

        max_iters (int): Maximum number of training iterations. Range: 100-100,000.
            Total number of times the model will see batches of training data.

        resolution (int): Input image resolution for the model. Range: 128-6400 pixels.
            Higher resolutions can improve accuracy but require more compute.

        wandb_project (Optional[str]): Weights & Biases project name in format "ORG_ID/PROJECT_NAME".
            Used for experiment tracking and visualization.

        wandb_apikey (Optional[str]): API key for Weights & Biases integration.
            Required if using wandb_project.

        learning_rate (float): Step size for model weight updates. Range: 0.00001-0.1.
            Controls how quickly the model learns. Too high can cause instability.

        decoder_multiplier (float): Multiplier for decoder learning rate.
            Allows different learning rates for decoder vs backbone.

        backbone_multiplier (float): Multiplier for backbone learning rate.
            Default 0.1 means backbone learns 10x slower than decoder.

        amp_enabled (bool): Whether to use automatic mixed precision training.
            Can speed up training and reduce memory usage with minimal accuracy impact.

        weight_decay (float): L2 regularization factor to prevent overfitting.
            Higher values = stronger regularization.

        ema_enabled (bool): Whether to use Exponential Moving Average of model weights.
            Can improve model stability and final performance.

        ema_decay (float): Decay rate for EMA. Higher = slower but more stable updates.
            Only used if ema_enabled=True.

        ema_warmup (int): Number of iterations before starting EMA.
            Only used if ema_enabled=True.

        freeze_bn (bool): Whether to freeze all batch normalization layers.
            Useful for fine-tuning with small batch sizes.

        freeze_bn_bkb (bool): Whether to freeze backbone batch normalization layers.
            Default True to preserve pretrained backbone statistics.

        optimizer (str): Optimization algorithm. Options: "ADAMW", "SGD", "RMSPROP".
            ADAMW generally works best for vision tasks.

        scheduler (str): Learning rate schedule. Options: "POLY", "FIXED", "COSINE", "MULTISTEP".
            Controls how learning rate changes during training.

        early_stop (bool): Whether to stop training early if validation metrics plateau.
            Can prevent overfitting and save compute time.

        patience (int): Number of evaluations to wait for improvement before early stopping.
            Only used if early_stop=True.

    Example:
    ```python
    from focoos import Focoos, Hyperparameters

    # Train with custom hyperparameters
    hyperparams = Hyperparameters(
        batch_size=8,  # Smaller batch size for limited GPU memory
        max_iters=2000,  # Train for more iterations
        learning_rate=1e-4,  # Lower learning rate for more stable training
        amp_enabled=True,  # Use mixed precision training
        ema_enabled=True,  # Use EMA for better stability
        early_stop=True,  # Enable early stopping
        patience=3,  # Stop after 3 evaluations without improvement
    )
    model.train(dataset_ref, hyperparams)
    ```

    """

    batch_size: Annotated[
        int,
        Field(
            ge=1,
            le=32,
            description="Batch size, how many images are processed at every iteration",
        ),
    ] = 16
    eval_period: Annotated[
        int,
        Field(ge=50, le=2000, description="How often iterations to evaluate the model"),
    ] = 500
    max_iters: Annotated[
        int,
        Field(1500, ge=100, le=100000, description="Maximum number of training iterations"),
    ] = 1500
    resolution: Annotated[int, Field(640, description="Model expected resolution", ge=128, le=6400)] = 640
    wandb_project: Annotated[
        Optional[str],
        Field(description="Wandb project name must be like ORG_ID/PROJECT_NAME"),
    ] = None
    wandb_apikey: Annotated[Optional[str], Field(description="Wandb API key")] = None
    learning_rate: Annotated[
        float,
        Field(gt=0.00001, lt=0.1, description="Learning rate"),
    ] = 5e-4
    decoder_multiplier: Annotated[float, Field(description="Backbone multiplier")] = 1
    backbone_multiplier: float = 0.1
    amp_enabled: Annotated[bool, Field(description="Enable automatic mixed precision")] = True
    weight_decay: Annotated[float, Field(description="Weight decay")] = 0.02
    ema_enabled: Annotated[bool, Field(description="Enable EMA (exponential moving average)")] = False
    ema_decay: Annotated[float, Field(description="EMA decay rate")] = 0.999
    ema_warmup: Annotated[int, Field(description="EMA warmup")] = 100
    freeze_bn: Annotated[bool, Field(description="Freeze batch normalization layers")] = False
    freeze_bn_bkb: Annotated[bool, Field(description="Freeze backbone batch normalization layers")] = True
    optimizer: Literal["ADAMW", "SGD", "RMSPROP"] = "ADAMW"
    scheduler: Literal["POLY", "FIXED", "COSINE", "MULTISTEP"] = "MULTISTEP"

    early_stop: Annotated[bool, Field(description="Enable early stopping")] = True
    patience: Annotated[
        int,
        Field(
            description="(Only with early_stop=True) Validation cycles after which the train is stopped if there's no improvement in accuracy."
        ),
    ] = 5

    @field_validator("wandb_project")
    def validate_wandb_project(cls, value):
        if value is not None:
            # Define a regex pattern to match valid characters
            if not re.match(r"^[\w.-/]+$", value):
                raise ValueError("Wandb project name must only contain characters, dashes, underscores, and dots.")
        return value


class TrainingInfo(FocoosBaseModel):
    """Information about a model's training process.

    Example:
    ```python
    from focoos import Focoos

    focoos = Focoos(api_key="<YOUR-API-KEY>")

    model = focoos.get_remote_model("my-model")

    info = model.train_info()
    print(f"Training status: {info.main_status}")
    print(f"Started at: {info.start_time}")
    print(f"Instance type: {info.instance_type}")
    print(f"Elapsed time: {info.elapsed_time} seconds")
    ```

    """

    algorithm_name: str
    instance_type: Optional[str] = None
    volume_size: Optional[int] = 100
    max_runtime_in_seconds: Optional[int] = 36000
    main_status: Optional[str] = None
    secondary_status: Optional[str] = None
    failure_reason: Optional[str] = None
    elapsed_time: Optional[int] = None
    status_transitions: list[dict] = []
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    artifact_location: Optional[str] = None


class ModelPreview(FocoosBaseModel):
    """Preview information for a Focoos model.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")

        # List all available models
        models = focoos.list_models()

        # Print info about each model
        for model in models:
            print(f"Model: {model.name}")
            print(f"Reference: {model.ref}")
            print(f"Task: {model.task}")
            print(f"Status: {model.status}")
            print(f"Description: {model.description}")
            print("---")
        ```
    """

    ref: str
    name: str
    task: FocoosTask
    description: Optional[str] = None
    status: ModelStatus
    focoos_model: str


class DatasetMetadata(FocoosBaseModel):
    """
    Metadata for a dataset.

    Example:
    ```python
    from focoos import Focoos

    focoos = Focoos(api_key="<YOUR-API-KEY>")

    # List all shared datasets
    datasets = focoos.list_shared_datasets()

    # Print info about each dataset
    for dataset in datasets:
        print(f"Dataset: {dataset.name}")
        print(f"Reference: {dataset.ref}")
        print(f"Task: {dataset.task}")
        print(f"Layout: {dataset.layout}")
        print(f"Description: {dataset.description}")
        print("---")
    ```
    """

    ref: str
    name: str
    task: FocoosTask
    layout: DatasetLayout
    description: Optional[str] = None


class ModelMetadata(FocoosBaseModel):
    """Complete metadata for a Focoos model.

    Example:
        ```python
        from focoos import Focoos, RemoteModel

        # Initialize Focoos client
        focoos = Focoos(api_key="<YOUR-API-KEY>")

        # Get a remote model instance
        model = focoos.get_remote_model("my-model")

        # Get model metadata
        metadata = model.get_info()

        print(f"Model: {metadata.name}")
        print(f"Reference: {metadata.ref}")
        print(f"Task: {metadata.task}")
        print(f"Status: {metadata.status}")
        print(f"Description: {metadata.description}")
        ```
    """

    ref: str
    name: str
    description: Optional[str] = None
    owner_ref: str
    focoos_model: str
    task: FocoosTask
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    metrics: Optional[dict] = None
    latencies: Optional[list[dict]] = None
    classes: Optional[list[str]] = None
    im_size: Optional[int] = None
    hyperparameters: Optional[Hyperparameters] = None
    training_info: Optional[TrainingInfo] = None
    location: Optional[str] = None
    dataset: Optional[DatasetMetadata] = None


class TrainInstance(str, Enum):
    """Available training instance types.

    Values:
        - ML_G4DN_XLARGE: ml.g4dn.xlarge instance
        - ML_G5_XLARGE: ml.g5.xlarge instance
        - ML_G5_12XLARGE: ml.g5.12xlarge instance
    """

    ML_G4DN_XLARGE = "ml.g4dn.xlarge"
    ML_G5_XLARGE = "ml.g5.xlarge"
    ML_G5_12XLARGE = "ml.g5.12xlarge"


class FocoosDet(FocoosBaseModel):
    """Single detection result from a model.

    Attributes:
        bbox (list[int]): Bounding box coordinates in xyxy absolute format.
        conf (float): Confidence score (from 0 to 1).
        cls_id (int): Class ID (0-indexed).
        label (str): Label (name of the class).
        mask (str): Mask (base64 encoded string having origin in the top left corner of bbox).

    !!! Note
        The mask is only present if the model is an instance segmentation or semantic segmentation model.
        The mask is a base64 encoded string having origin in the top left corner of bbox and the same width and height of the bbox.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")

        # Get a remote model instance
        model = focoos.get_remote_model("my-model")

        # Run inference on an image
        image_path = "image.jpg"
        results, _ = model.infer(image_path)

        # Print detection results
        for det in results.detections:
            print(f"Found {det.label} with confidence {det.conf:.2f}")
            print(f"Bounding box: {det.bbox}")
            if det.mask:
                print("Instance segmentation mask included")
        ```
    """

    bbox: Optional[list[int]] = None
    conf: Optional[float] = None
    cls_id: Optional[int] = None
    label: Optional[str] = None
    mask: Optional[str] = None

    @classmethod
    def from_json(cls, data: Union[str, dict]):
        if isinstance(data, str):
            with open(data, encoding="utf-8") as f:
                data_dict = json.load(f)
        else:
            data_dict = data

        bbox = data_dict.get("bbox")
        if bbox is not None:  # Retrocompatibility fix for remote results with float bbox, !TODO remove asap
            data_dict["bbox"] = list(map(int, bbox))

        return cls.model_validate(data_dict)


class FocoosDetections(FocoosBaseModel):
    """Collection of detection results from a model.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")

        # Get a remote model instance
        model = focoos.get_remote_model("my-model")

        # Run inference on an image
        image_path = "image.jpg"
        results, _ = model.infer(image_path)

        # Print detection results
        for det in results.detections:
            print(f"Found {det.label} with confidence {det.conf:.2f}")
            print(f"Bounding box: {det.bbox}")
            if det.mask:
                print("Instance segmentation mask included")
        ```
    """

    detections: list[FocoosDet]
    latency: Optional[dict] = None


@dataclass
class OnnxRuntimeOpts:
    """ONNX runtime configuration options.

    Example:
        ```python
        from focoos import ONNXRuntime, FocoosTask, ModelMetadata

        # Configure ONNX Runtime options
        opts = OnnxRuntimeOpts(
            fp16=True,  # Enable FP16 precision
            cuda=True,  # Use CUDA execution provider
            trt=False,  # Disable TensorRT
            vino=False,  # Disable OpenVINO
            coreml=False,  # Disable CoreML
            warmup_iter=10,  # Number of warmup iterations
            verbose=False,  # Disable verbose logging
        )

        # Create ONNX Runtime instance
        model_path = "model.onnx"
        model_metadata = ModelMetadata(task=FocoosTask.DETECTION)
        runtime = ONNXRuntime(model_path, opts, model_metadata)
        ```
    """

    fp16: Optional[bool] = False
    cuda: Optional[bool] = False
    vino: Optional[bool] = False
    verbose: Optional[bool] = False
    trt: Optional[bool] = False
    coreml: Optional[bool] = False
    warmup_iter: int = 0


@dataclass
class TorchscriptRuntimeOpts:
    """TorchScript runtime configuration options.

    Example:
        ```python
        from focoos import TorchscriptRuntime, FocoosTask, ModelMetadata

        # Configure TorchScript Runtime options
        opts = TorchscriptRuntimeOpts(
            warmup_iter=10,  # Number of warmup iterations
            optimize_for_inference=True,  # Enable inference optimizations
            set_fusion_strategy=True,  # Enable operator fusion
        )

        # Create TorchScript Runtime instance
        model_path = "model.pt"
        model_metadata = ModelMetadata(task=FocoosTask.DETECTION)
        runtime = TorchscriptRuntime(model_path, opts, model_metadata)
        ```
    """

    warmup_iter: int = 0
    optimize_for_inference: bool = True
    set_fusion_strategy: bool = True


@dataclass
class LatencyMetrics:
    """Performance metrics for model inference.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")

        # Load model and run benchmark
        model = focoos.get_local_model("my-model")
        metrics = model.benchmark(iterations=20, size=640)

        # Access latency metrics
        print(f"FPS: {metrics.fps}")
        print(f"Mean latency: {metrics.mean} ms")
        print(f"Engine: {metrics.engine}")
        print(f"Device: {metrics.device}")
        print(f"Input size: {metrics.im_size}x{metrics.im_size}")
        ```
    """

    fps: int
    engine: str
    min: float
    max: float
    mean: float
    std: float
    im_size: int
    device: str


class RuntimeTypes(str, Enum):
    """Available runtime configurations for model inference.

    Values:
        - ONNX_CUDA32: ONNX with CUDA FP32
        - ONNX_TRT32: ONNX with TensorRT FP32
        - ONNX_TRT16: ONNX with TensorRT FP16
        - ONNX_CPU: ONNX on CPU
        - ONNX_COREML: ONNX with CoreML
        - TORCHSCRIPT_32: TorchScript FP32

    """

    ONNX_CUDA32 = "onnx_cuda32"
    ONNX_TRT32 = "onnx_trt32"
    ONNX_TRT16 = "onnx_trt16"
    ONNX_CPU = "onnx_cpu"
    ONNX_COREML = "onnx_coreml"
    TORCHSCRIPT_32 = "torchscript_32"


class ModelFormat(str, Enum):
    """Supported model formats.

    Values:
        - ONNX: ONNX format
        - TORCHSCRIPT: TorchScript format

    """

    ONNX = "onnx"
    TORCHSCRIPT = "pt"

    @classmethod
    def from_runtime_type(cls, runtime_type: RuntimeTypes):
        if runtime_type in [
            RuntimeTypes.ONNX_CUDA32,
            RuntimeTypes.ONNX_TRT32,
            RuntimeTypes.ONNX_TRT16,
            RuntimeTypes.ONNX_CPU,
            RuntimeTypes.ONNX_COREML,
        ]:
            return cls.ONNX
        elif runtime_type == RuntimeTypes.TORCHSCRIPT_32:
            return cls.TORCHSCRIPT
        else:
            raise ValueError(f"Invalid runtime type: {runtime_type}")


class GPUInfo(FocoosBaseModel):
    """Information about a GPU device.

    ```
    """

    gpu_id: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_used_percentage: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_load_percentage: Optional[float] = None


class SystemInfo(FocoosBaseModel):
    """System information including hardware and software details."""

    focoos_host: Optional[str] = None
    system: Optional[str] = None
    system_name: Optional[str] = None
    cpu_type: Optional[str] = None
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    memory_used_percentage: Optional[float] = None
    available_providers: Optional[list[str]] = None
    disk_space_total_gb: Optional[float] = None
    disk_space_used_percentage: Optional[float] = None
    gpu_count: Optional[int] = None
    gpu_driver: Optional[str] = None
    gpu_cuda_version: Optional[str] = None
    gpus_info: Optional[list[GPUInfo]] = None
    packages_versions: Optional[dict[str, str]] = None
    environment: Optional[dict[str, str]] = None

    def pretty_print(self):
        print("================ SYSTEM INFO ====================")
        for key, value in self.model_dump().items():
            if isinstance(value, list):
                print(f"{key}:")
                if key == "gpus_info":  # Special formatting for gpus_info.
                    for item in value:
                        print(f"- id: {item['gpu_id']}")
                        for sub_key, sub_value in item.items():
                            if sub_key != "gpu_id" and sub_value is not None:
                                formatted_key = sub_key.replace("_", "-")
                                print(f"    - {formatted_key}: {sub_value}")
                else:
                    for item in value:
                        print(f"  - {item}")
            elif isinstance(value, dict) and key == "packages_versions":  # Special formatting for packages_versions
                print(f"{key}:")
                for pkg_name, pkg_version in value.items():
                    print(f"  - {pkg_name}: {pkg_version}")
            elif isinstance(value, dict) and key == "environment":  # Special formatting for environment
                print(f"{key}:")
                for env_key, env_value in value.items():
                    print(f"  - {env_key}: {env_value}")
            else:
                print(f"{key}: {value}")
        print("================================================")


class ApiKey(FocoosBaseModel):
    """API key for authentication."""

    key: str  # type: ignore


class Quotas(FocoosBaseModel):
    """Usage quotas and limits for a user account.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")
        user_info = focoos.get_user_info()

        # Access quotas from user info
        quotas = user_info.quotas
        print(f"Total inferences: {quotas.total_inferences}")
        print(f"Max inferences: {quotas.max_inferences}")
        print(f"Used storage (GB): {quotas.used_storage_gb}")
        print(f"Max storage (GB): {quotas.max_storage_gb}")
        print(f"Active training jobs: {quotas.active_training_jobs}")
        print(f"Max active training jobs: {quotas.max_active_training_jobs}")
        print(f"Used MLG4DNXLarge training jobs hours: {quotas.used_mlg4dnxlarge_training_jobs_hours}")
        print(f"Max MLG4DNXLarge training jobs hours: {quotas.max_mlg4dnxlarge_training_jobs_hours}")
        ```
    """

    # INFERENCE
    total_inferences: int
    max_inferences: int
    # STORAGE
    used_storage_gb: float
    max_storage_gb: float
    # TRAINING
    active_training_jobs: list[str]
    max_active_training_jobs: int

    # ML_G4DN_XLARGE TRAINING HOURS
    used_mlg4dnxlarge_training_jobs_hours: float
    max_mlg4dnxlarge_training_jobs_hours: float


class User(FocoosBaseModel):
    """User account information.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")
        user_info = focoos.get_user_info()

        # Access user info fields
        print(f"Email: {user_info.email}")
        print(f"Created at: {user_info.created_at}")
        print(f"Updated at: {user_info.updated_at}")
        print(f"Company: {user_info.company}")
        print(f"API key: {user_info.api_key.key}")

        # Access quotas
        quotas = user_info.quotas
        print(f"Total inferences: {quotas.total_inferences}")
        print(f"Max inferences: {quotas.max_inferences}")
        ```
    """

    email: str
    created_at: datetime
    updated_at: datetime
    company: Optional[str] = None
    api_key: ApiKey
    quotas: Quotas


class ModelNotFound(Exception):
    """Exception raised when a requested model is not found."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class Metrics(FocoosBaseModel):
    """Collection of training and inference metrics.

    Example:
        ```python
        from focoos import Focoos

        focoos = Focoos(api_key="<YOUR-API-KEY>")
        model = focoos.get_remote_model("my-model")

        # Get model metrics
        metrics = model.metrics()

        # Access metrics fields
        print(f"Inference metrics: {metrics.infer_metrics}")
        print(f"Validation metrics: {metrics.valid_metrics}")
        print(f"Training metrics: {metrics.train_metrics}")
        print(f"Total iterations: {metrics.iterations}")
        print(f"Best validation: {metrics.best_valid_metric}")
        print(f"Last updated: {metrics.updated_at}")
        ```
    """

    infer_metrics: list[dict] = []
    valid_metrics: list[dict] = []
    train_metrics: list[dict] = []
    iterations: Optional[int] = None
    best_valid_metric: Optional[dict] = None
    updated_at: Optional[datetime] = None
