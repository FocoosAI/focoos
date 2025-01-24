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


class DeploymentMode(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class ModelStatus(str, Enum):
    CREATED = "CREATED"
    TRAINING_STARTING = "TRAINING_STARTING"
    TRAINING_RUNNING = "TRAINING_RUNNING"
    TRAINING_ERROR = "TRAINING_ERROR"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    TRAINING_STOPPED = "TRAINING_STOPPED"
    DEPLOYED = "DEPLOYED"
    DEPLOY_ERROR = "DEPLOY_ERROR"


class DatasetLayout(str, Enum):
    ROBOFLOW_COCO = "roboflow_coco"
    ROBOFLOW_SEG = "roboflow_seg"
    CATALOG = "catalog"
    SUPERVISELY = "supervisely"


class FocoosTask(str, Enum):
    DETECTION = "detection"
    SEMSEG = "semseg"
    INSTANCE_SEGMENTATION = "instseg"


class Hyperparameters(FocoosBaseModel):
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
    ref: str
    name: str
    task: FocoosTask
    description: Optional[str] = None
    status: ModelStatus
    focoos_model: str


class DatasetInfo(FocoosBaseModel):
    url: Annotated[
        str,
        Field(
            description="🗂️ Dataset url to use for the project, must be a valid S3 URL",
        ),
    ]
    name: Annotated[
        str,
        Field(
            description="🗂️ Dataset name",
        ),
    ]
    layout: Annotated[
        DatasetLayout,
        Field(
            default=None,
            description="🗂️ Dataset layout, can be any of the following: "
            + ", ".join([layout.value for layout in DatasetLayout]),
        ),
    ]

    @field_validator("url")
    def validate_s3_url(cls, v: str):
        if not S3_URL_REGEX.match(v):
            raise ValueError("Invalid S3 URL format, must be s3://BUCKET_NAME/path")
        return v


class ModelMetadata(FocoosBaseModel):
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
    dataset: Optional[DatasetInfo] = None


class DatasetMetadata(FocoosBaseModel):
    ref: str
    name: str
    task: FocoosTask
    layout: DatasetLayout
    description: Optional[str] = None


class TrainInstance(str, Enum):
    ML_G4DN_XLARGE = "ml.g4dn.xlarge"
    ML_G5_XLARGE = "ml.g5.xlarge"
    ML_G5_12XLARGE = "ml.g5.12xlarge"


class FocoosDet(FocoosBaseModel):
    bbox: Optional[list[float]] = None
    conf: Optional[float] = None
    cls_id: Optional[int] = None
    label: Optional[str] = None
    mask: Optional[str] = None


class FocoosDetections(FocoosBaseModel):
    detections: list[FocoosDet]
    latency: Optional[dict] = None


@dataclass
class OnnxEngineOpts:
    fp16: Optional[bool] = False
    cuda: Optional[bool] = False
    vino: Optional[bool] = False
    verbose: Optional[bool] = False
    trt: Optional[bool] = False
    coreml: Optional[bool] = False
    warmup_iter: int = 0


@dataclass
class LatencyMetrics:
    fps: int
    engine: str
    min: float
    max: float
    mean: float
    std: float
    im_size: int
    device: str


class RuntimeTypes(str, Enum):
    ONNX_CUDA32 = "onnx_cuda32"
    ONNX_TRT32 = "onnx_trt32"
    ONNX_TRT16 = "onnx_trt16"
    ONNX_CPU = "onnx_cpu"
    ONNX_COREML = "onnx_coreml"


class GPUInfo(FocoosBaseModel):
    gpu_id: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_used_percentage: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_load_percentage: Optional[float] = None


class SystemInfo(FocoosBaseModel):
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
            else:
                print(f"{key}: {value}")
        print("================================================")


class ApiKey(FocoosBaseModel):
    key: str  # type: ignore


class Quotas(FocoosBaseModel):
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
    email: str
    created_at: datetime
    updated_at: datetime
    company: Optional[str] = None
    api_key: ApiKey
    quotas: Quotas


class ModelNotFound(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class Metrics(FocoosBaseModel):
    infer_metrics: list[dict] = []
    valid_metrics: list[dict] = []
    train_metrics: list[dict] = []
    iterations: Optional[int] = None
    best_valid_metric: Optional[dict] = None
    updated_at: Optional[datetime] = None
