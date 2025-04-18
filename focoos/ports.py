import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, field_validator

from focoos.structures import Instances

S3_URL_REGEX = re.compile(r"^s3://" r"(?P<bucket>[a-zA-Z0-9.-]+)/" r"(?P<path>.+(\.tar\.gz|\.zip)?)$")

DEV_API_URL = "https://api.dev.focoos.ai/v0"
PROD_API_URL = "https://api.focoos.ai/v0"
LOCAL_API_URL = "http://localhost:8501/v0"

ROOT_DIR = Path.home() / ".cache" / "focoos"
ROOT_DIR = str(ROOT_DIR) if os.name == "nt" else ROOT_DIR
MODELS_ROOT = os.path.join(ROOT_DIR, "models")
DATASETS_ROOT = os.path.join(ROOT_DIR, "datasets")


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
        - ROBOFLOW_COCO: (Detection,Instance Segmentation)
        - ROBOFLOW_SEG: (Semantic Segmentation)
        - SUPERVISELY: (Semantic Segmentation)
    Example:
        ```python
        - ROBOFLOW_COCO: (Detection,Instance Segmentation) Roboflow COCO format:
            root/
                train/
                    - _annotations.coco.json
                    - img_1.jpg
                    - img_2.jpg
                valid/
                    - _annotations.coco.json
                    - img_3.jpg
                    - img_4.jpg
        - ROBOFLOW_SEG: (Semantic Segmentation) Roboflow segmentation format:
            root/
                train/
                    - _classes.csv (comma separated csv)
                    - img_1.jpg
                    - img_2.jpg
                valid/
                    - _classes.csv (comma separated csv)
                    - img_3_mask.png
                    - img_4_mask.png

        - SUPERVISELY: (Semantic Segmentation) format:
            root/
                train/
                    meta.json
                    img/
                    ann/
                    mask/
                valid/
                    meta.json
                    img/
                    ann/
                    mask/
        ```
    """

    ROBOFLOW_COCO = "roboflow_coco"
    ROBOFLOW_SEG = "roboflow_seg"
    CATALOG = "catalog"
    SUPERVISELY = "supervisely"


class Task(str, Enum):
    """Types of computer vision tasks supported by Focoos.

    Values:
        - DETECTION: Object detection
        - SEMSEG: Semantic segmentation
        - INSTANCE_SEGMENTATION: Instance segmentation
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

    This class contains details about the training job configuration, status, and timing.

    Attributes:
        algorithm_name: The name of the training algorithm used.
        instance_type: The compute instance type used for training.
        volume_size: The storage volume size in GB allocated for the training job.
        max_runtime_in_seconds: Maximum allowed runtime for the training job in seconds.
        main_status: The primary status of the training job (e.g., "InProgress", "Completed").
        secondary_status: Additional status information about the training job.
        failure_reason: Description of why the training job failed, if applicable.
        elapsed_time: Time elapsed since the start of the training job in seconds.
        status_transitions: List of status change events during the training process.
        start_time: Timestamp when the training job started.
        end_time: Timestamp when the training job completed or failed.
        artifact_location: Storage location of the training artifacts and model outputs.
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

    This class provides a lightweight preview of model information in the Focoos platform,
    containing essential details like reference ID, name, task type, and status.

    Attributes:
        ref (str): Unique reference ID for the model.
        name (str): Human-readable name of the model.
        task (FocoosTask): The computer vision task this model is designed for.
        description (Optional[str]): Optional description of the model's purpose or capabilities.
        status (ModelStatus): Current status of the model (e.g., training, ready, failed).
        focoos_model (str): The base model architecture identifier.
    """

    ref: str
    name: str
    task: Task
    description: Optional[str] = None
    status: ModelStatus
    focoos_model: str


class DatasetSpec(FocoosBaseModel):
    """Specification details for a dataset in the Focoos platform.

    This class provides information about the dataset's size and composition,
    including the number of samples in training and validation sets and the total size.

    Attributes:
        train_length (int): Number of samples in the training set.
        valid_length (int): Number of samples in the validation set.
        size_mb (float): Total size of the dataset in megabytes.
    """

    train_length: int
    valid_length: int
    size_mb: float


class DatasetPreview(FocoosBaseModel):
    """Preview information for a Focoos dataset.

    This class provides metadata about a dataset in the Focoos platform,
    including its identification, task type, and layout format.

    Attributes:
        ref (str): Unique reference ID for the dataset.
        name (str): Human-readable name of the dataset.
        task (FocoosTask): The computer vision task this dataset is designed for.
        layout (DatasetLayout): The structural format of the dataset (e.g., ROBOFLOW_COCO, ROBOFLOW_SEG, SUPERVISELY).
        description (Optional[str]): Optional description of the dataset's purpose or contents.
        spec (Optional[DatasetSpec]): Detailed specifications about the dataset's composition and size.
    """

    ref: str
    name: str
    task: Task
    layout: DatasetLayout
    description: Optional[str] = None
    spec: Optional[DatasetSpec] = None


class RemoteModelInfo(FocoosBaseModel):
    """Complete metadata for a Focoos model.

    This class contains comprehensive information about a model in the Focoos platform,
    including its identification, configuration, performance metrics, and training details.

    Attributes:
        ref (str): Unique reference ID for the model.
        name (str): Human-readable name of the model.
        description (Optional[str]): Optional description of the model's purpose or capabilities.
        owner_ref (str): Reference ID of the model owner.
        focoos_model (str): The base model architecture used.
        task (FocoosTask): The task type the model is designed for (e.g., DETECTION, SEMSEG).
        created_at (datetime): Timestamp when the model was created.
        updated_at (datetime): Timestamp when the model was last updated.
        status (ModelStatus): Current status of the model (e.g., TRAINING, DEPLOYED).
        metrics (Optional[dict]): Performance metrics of the model (e.g., mAP, accuracy).
        latencies (Optional[list[dict]]): Inference latency measurements across different configurations.
        classes (Optional[list[str]]): List of class names the model can detect or segment.
        im_size (Optional[int]): Input image size the model expects.
        hyperparameters (Optional[Hyperparameters]): Training hyperparameters used.
        training_info (Optional[TrainingInfo]): Information about the training process.
        location (Optional[str]): Storage location of the model.
        dataset (Optional[DatasetPreview]): Information about the dataset used for training.
    """

    ref: str
    name: str
    description: Optional[str] = None
    owner_ref: str
    focoos_model: str
    task: Task
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
    dataset: Optional[DatasetPreview] = None


class TrainInstance(str, Enum):
    """Available training instance types.

    Values:
        - ML_G4DN_XLARGE: ml.g4dn.xlarge instance, Nvidia Tesla T4, 16GB RAM, 4vCPU
    """

    ML_G4DN_XLARGE = "ml.g4dn.xlarge"


class FocoosDet(FocoosBaseModel):
    """Single detection result from a model.

    This class represents a single detection or segmentation result from a Focoos model.
    It contains information about the detected object including its position, class,
    confidence score, and optional segmentation mask.

    Attributes:
        bbox (Optional[list[int]]): Bounding box coordinates in [x1, y1, x2, y2] format,
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
        conf (Optional[float]): Confidence score of the detection, ranging from 0 to 1.
        cls_id (Optional[int]): Class ID of the detected object, corresponding to the index
            in the model's class list.
        label (Optional[str]): Human-readable label of the detected object.
        mask (Optional[str]): Base64-encoded PNG image representing the segmentation mask.
            Note that the mask is cropped to the bounding box coordinates and does not
            have the same shape as the input image.

    !!! Note
        The mask is only present if the model is an instance segmentation or semantic segmentation model.
        The mask is a base64 encoded string having origin in the top left corner of bbox and the same width and height of the bbox.

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

    This class represents a collection of detection or segmentation results from a Focoos model.
    It contains a list of individual detections and optional latency information.

    Attributes:
        detections (list[FocoosDet]): List of detection results, where each detection contains
            information about a detected object including its position, class, confidence score,
            and optional segmentation mask.
        latency (Optional[dict]): Dictionary containing latency information for the inference process.
            Typically includes keys like 'inference', 'preprocess', and 'postprocess' with values
            representing the time taken in seconds for each step.
    """

    detections: list[FocoosDet]
    latency: Optional[dict] = None


@dataclass
class OnnxRuntimeOpts:
    """ONNX runtime configuration options.

    This class provides configuration options for the ONNX runtime used for model inference.

    Attributes:
        fp16 (Optional[bool]): Enable FP16 precision. Default is False.
        cuda (Optional[bool]): Enable CUDA acceleration for GPU inference. Default is False.
        vino (Optional[bool]): Enable OpenVINO acceleration for Intel hardware. Default is False.
        verbose (Optional[bool]): Enable verbose logging during inference. Default is False.
        trt (Optional[bool]): Enable TensorRT acceleration for NVIDIA GPUs. Default is False.
        coreml (Optional[bool]): Enable CoreML acceleration for Apple hardware. Default is False.
        warmup_iter (int): Number of warmup iterations to run before benchmarking. Default is 0.

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

    This class provides configuration options for the TorchScript runtime used for model inference.

    Attributes:
        warmup_iter (int): Number of warmup iterations to run before benchmarking. Default is 0.
        optimize_for_inference (bool): Enable inference optimizations. Default is True.
        set_fusion_strategy (bool): Enable operator fusion. Default is True.
    """

    warmup_iter: int = 0
    optimize_for_inference: bool = True
    set_fusion_strategy: bool = True


@dataclass
class LatencyMetrics:
    """Performance metrics for model inference.

    This class provides performance metrics for model inference, including frames per second (FPS),
    engine used, minimum latency, maximum latency, mean latency, standard deviation of latency,
    input image size, and device type.

    Attributes:
        fps (int): Frames per second (FPS) of the inference process.
        engine (str): The inference engine used (e.g., "onnx", "torchscript").
        min (float): Minimum latency in milliseconds.
        max (float): Maximum latency in milliseconds.
        mean (float): Mean latency in milliseconds.
        std (float): Standard deviation of latency in milliseconds.
        im_size (int): Input image size.
        device (str): Device type.
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


class GPUDevice(FocoosBaseModel):
    """Information about a GPU device."""

    gpu_id: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_used_percentage: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_load_percentage: Optional[float] = None


class GPUInfo(FocoosBaseModel):
    """Information about a GPU driver."""

    gpu_count: Optional[int] = None
    gpu_driver: Optional[str] = None
    gpu_cuda_version: Optional[str] = None
    devices: Optional[list[GPUDevice]] = None


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
    gpu_info: Optional[GPUInfo] = None
    packages_versions: Optional[dict[str, str]] = None
    environment: Optional[dict[str, str]] = None

    def pretty_print(self):
        print("================ SYSTEM INFO ====================")
        for key, value in self.model_dump().items():
            if key == "gpu_info" and value is not None:
                print(f"{key}:")
                print(f"  - gpu_count: {value.get('gpu_count')}")
                print(f"  - gpu_driver: {value.get('gpu_driver')}")
                print(f"  - gpu_cuda_version: {value.get('gpu_cuda_version')}")
                if value.get("devices"):
                    print("  - devices:")
                    for device in value.get("devices", []):
                        print(f"    - GPU {device.get('gpu_id')}:")
                        for device_key, device_value in device.items():
                            if device_key != "gpu_id":
                                print(f"      - {device_key}: {device_value}")
            elif isinstance(value, list):
                print(f"{key}:")
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

    Attributes:
        total_inferences (int): Total number of inferences allowed.
        max_inferences (int): Maximum number of inferences allowed.
        used_storage_gb (float): Used storage in gigabytes.
        max_storage_gb (float): Maximum storage in gigabytes.
        active_training_jobs (list[str]): List of active training job IDs.
        max_active_training_jobs (int): Maximum number of active training jobs allowed.
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

    This class represents a user account in the Focoos platform, containing
    personal information, API key, and usage quotas.

    Attributes:
        email (str): The user's email address.
        created_at (datetime): When the user account was created.
        updated_at (datetime): When the user account was last updated.
        company (Optional[str]): The user's company name, if provided.
        api_key (ApiKey): The API key associated with the user account.
        quotas (Quotas): Usage quotas and limits for the user account.
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
    """
    Collection of training and inference metrics.
    """

    infer_metrics: list[dict] = []
    valid_metrics: list[dict] = []
    train_metrics: list[dict] = []
    iterations: Optional[int] = None
    best_valid_metric: Optional[dict] = None
    updated_at: Optional[datetime] = None


class ModelFamily(str, Enum):
    """Enumerazione delle famiglie di modelli disponibili"""

    RTDETR = "fai_rtdetr"
    M2F = "fai_m2f"
    PEM = "fai_pem"
    BF = "fai_bf"


@dataclass
class ModelConfig:
    num_classes: int
    # other parameters are model-specific


@dataclass
class ModelOutput:
    def to_instances(self) -> Optional[Instances]:
        pass


class DatasetSplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class ModelInfo:
    """Detailed information about a specific model.

    This class stores all the necessary information to identify, configure, and evaluate a model.

    Attributes:
        name: Unique identifier for the model.
        model_family: The family/architecture the model belongs to (e.g., RTDETR, M2F).
        config_class: The configuration class type used to instantiate the model.
        classes: List of class names the model can detect/segment.
        im_size: Input image size (typically square dimensions).
        task: The task the model performs (detection, segmentation, etc.).
        config: Configuration instance with model-specific parameters.
        description: Optional human-readable description of the model.
        weights_uri: Optional path or URI to the model weights.
        val_dataset: Optional name of the validation dataset used.
        val_metrics: Optional dictionary containing validation metrics.
        latency: Optional list of latency measurements across different runtimes.
    """

    name: str
    model_family: ModelFamily
    config_class: Type[ModelConfig]
    classes: list[str]
    im_size: int
    task: Task
    config: ModelConfig
    description: Optional[str] = None
    weights_uri: Optional[str] = None
    val_dataset: Optional[str] = None
    val_metrics: Optional[dict] = None  # todo: make them explicit
    latency: Optional[list[LatencyMetrics]] = None

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            model_info = json.load(f)
        return cls(**model_info)

    def dump_json(self, path: str):
        data = asdict(self)
        # Convert config_class to string
        data["config_class"] = self.config_class.__name__
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


@dataclass
class DatasetMetadata:
    """Dataclass for storing dataset metadata."""

    num_classes: int
    task: Task
    count: Optional[int] = None
    name: Optional[str] = None
    image_root: Optional[str] = None
    thing_classes: Optional[List[str]] = None
    _thing_colors: Optional[List[Tuple]] = None
    stuff_classes: Optional[List[str]] = None
    _stuff_colors: Optional[List[Tuple]] = None
    sem_seg_root: Optional[str] = None
    panoptic_root: Optional[str] = None
    ignore_label: Optional[int] = None
    thing_dataset_id_to_contiguous_id: Optional[dict] = None
    stuff_dataset_id_to_contiguous_id: Optional[dict] = None
    json_file: Optional[str] = None

    @property
    def classes(self):  #!TODO: check if this is correct
        if self.task == Task.DET or self.task == Task.INSTSEG:
            return self.thing_classes
        if self.task == Task.SEMSEG or self.task == Task.PANSEG:
            # fixme: not sure for panoptic
            return self.stuff_classes

    @property
    def stuff_colors(self):
        if self._stuff_colors is not None:
            return self._stuff_colors
        if self.stuff_classes is None:
            return []
        return [((i * 64) % 255, (i * 128) % 255, (i * 32) % 255) for i in range(len(self.stuff_classes))]

    @stuff_colors.setter
    def stuff_colors(self, colors):
        self._stuff_colors = colors

    @property
    def thing_colors(self):
        if self._thing_colors is not None:
            return self._thing_colors
        if self.thing_classes is None:
            return []
        return [((i * 64) % 255, (i * 128) % 255, (i * 32) % 255) for i in range(1, len(self.thing_classes) + 1)]

    @thing_colors.setter
    def thing_colors(self, colors):
        self._thing_colors = colors

    @classmethod
    def from_dict(cls, metadata: dict):
        """Create DatasetMetadata from a dictionary.

        Args:
            metadata (dict): Dictionary containing metadata.

        Returns:
            DatasetMetadata: Instance of DatasetMetadata.
        """
        metadata = {k: v for k, v in metadata.items() if k in inspect.signature(cls).parameters}
        metadata["task"] = Task(metadata["task"])
        return cls(**metadata)

    @classmethod
    def from_json(cls, path: str):
        """Create DatasetMetadata from a json file.

        Args:
            path (str): Path to json file.

        Returns:
            DatasetMetadata: Instance of DatasetMetadata.
        """
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["task"] = Task(metadata["task"])
        return cls(**metadata)

    def dump_json(self, path: str):
        """Dump DatasetMetadata to a json file.

        Args:
            path (str): Path to json file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

    def get(self, attr, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return default


@dataclass
class TrainerArgs:
    """Configuration class for unified model training.

    Attributes:
        run_name (str): Name of the training run
        output_dir (str): Directory to save outputs
        ckpt_dir (Optional[str]): Directory for checkpoints
        init_checkpoint (Optional[str]): Initial checkpoint to load
        resume (bool): Whether to resume from checkpoint
        num_gpus (int): Number of GPUs to use
        device (str): Device to use (cuda/cpu)
        workers (int): Number of data loading workers
        amp_enabled (bool): Whether to use automatic mixed precision
        ddp_broadcast_buffers (bool): Whether to broadcast buffers in DDP
        ddp_find_unused (bool): Whether to find unused parameters in DDP
        checkpointer_period (int): How often to save checkpoints
        checkpointer_max_to_keep (int): Maximum checkpoints to keep
        eval_period (int): How often to evaluate
        log_period (int): How often to log
        vis_period (int): How often to visualize
        samples (int): Number of samples for visualization
        seed (int): Random seed
        early_stop (bool): Whether to use early stopping
        patience (int): Early stopping patience
        ema_enabled (bool): Whether to use EMA
        ema_decay (float): EMA decay rate
        ema_warmup (int): EMA warmup period
        learning_rate (float): Base learning rate
        weight_decay (float): Weight decay
        max_iters (int): Maximum training iterations
        batch_size (int): Batch size
        scheduler (str): Learning rate scheduler type
        scheduler_extra (Optional[dict]): Extra scheduler parameters
        optimizer (str): Optimizer type
        optimizer_extra (Optional[dict]): Extra optimizer parameters
        weight_decay_norm (float): Weight decay for normalization layers
        weight_decay_embed (float): Weight decay for embeddings
        backbone_multiplier (float): Learning rate multiplier for backbone
        decoder_multiplier (float): Learning rate multiplier for decoder
        head_multiplier (float): Learning rate multiplier for head
        freeze_bn (bool): Whether to freeze batch norm
        freeze_bn_bkb (bool): Whether to freeze backbone batch norm
        reset_classifier (bool): Whether to reset classifier
        clip_gradients (float): Gradient clipping value
        size_divisibility (int): Input size divisibility requirement
        gather_metric_period (int): How often to gather metrics
        zero_grad_before_forward (bool): Whether to zero gradients before forward pass
    """

    run_name: str
    output_dir: str
    ckpt_dir: Optional[str] = None
    init_checkpoint: Optional[str] = None
    resume: bool = False
    # Logistics params
    num_gpus: Optional[int] = 0
    device: str = "cuda"
    workers: int = 4
    amp_enabled: bool = True
    ddp_broadcast_buffers: bool = False
    ddp_find_unused: bool = True
    checkpointer_period: int = 1000
    checkpointer_max_to_keep: int = 1
    eval_period: int = 50
    log_period: int = 20
    vis_period: int = 5000
    samples: int = 4
    seed: int = 42
    early_stop: bool = False
    patience: int = 10
    # EMA
    ema_enabled: bool = False
    ema_decay: float = 0.999
    ema_warmup: int = 2000
    # Hyperparameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.02
    max_iters: int = 3000
    batch_size: int = 16
    scheduler: str = "POLY"
    scheduler_extra: Optional[dict] = None
    optimizer: str = "AdamW"
    optimizer_extra: Optional[dict] = None
    weight_decay_norm: float = 0.0
    weight_decay_embed: float = 0.0
    backbone_multiplier: float = 0.1
    decoder_multiplier: float = 1.0
    head_multiplier: float = 1.0
    freeze_bn: bool = False
    freeze_bn_bkb: bool = False
    reset_classifier: bool = False
    clip_gradients: float = 0.1
    size_divisibility: int = 0
    # Training specific
    gather_metric_period: int = 1
    zero_grad_before_forward: bool = False


@dataclass
class DetectronDict:
    file_name: str
    height: Optional[int] = None
    width: Optional[int] = None
    image_id: Optional[Union[str, int]] = None
    sem_seg_file_name: Optional[str] = None
    pan_seg_file_name: Optional[str] = None
    annotations: Optional[list[dict]] = None
    segments_info: Optional[list[dict]] = None
