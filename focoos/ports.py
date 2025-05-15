import inspect
import json
import os
from abc import ABC
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel
from torch import Tensor

from focoos.structures import Instances

DEV_API_URL = "https://api.dev.focoos.ai/v0"
PROD_API_URL = "https://api.focoos.ai/v0"
LOCAL_API_URL = "http://localhost:8501/v0"


ROOT_DIR = Path.home() / "FocoosAI"
ROOT_DIR = str(ROOT_DIR) if os.name == "nt" else ROOT_DIR
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


class PydanticBase(BaseModel, ABC):
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
    CLS_FOLDER = "cls_folder"


class Task(str, Enum):
    """Types of computer vision tasks supported by Focoos.

    Values:
        - DETECTION: Object detection
        - SEMSEG: Semantic segmentation
        - INSTANCE_SEGMENTATION: Instance segmentation
        - CLASSIFICATION: Image classification
    """

    DETECTION = "detection"
    SEMSEG = "semseg"
    INSTANCE_SEGMENTATION = "instseg"
    CLASSIFICATION = "classification"


@dataclass
class StatusTransition:
    status: ModelStatus
    timestamp: str
    iter: Optional[int] = None
    detail: Optional[str] = None


@dataclass
class TrainingInfo:
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

    instance_device: Optional[str] = None
    instance_type: Optional[str] = None
    volume_size: Optional[int] = None
    main_status: Optional[str] = None
    failure_reason: Optional[str] = None
    status_transitions: Optional[list[StatusTransition]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    artifact_location: Optional[str] = None


class ModelPreview(PydanticBase):
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


class DatasetSpec(PydanticBase):
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


class DatasetPreview(PydanticBase):
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


class RemoteModelInfo(PydanticBase):
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
    training_info: Optional[TrainingInfo] = None
    location: Optional[str] = None
    dataset: Optional[DatasetPreview] = None


class FocoosDet(PydanticBase):
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


class FocoosDetections(PydanticBase):
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

    def __len__(self):
        return len(self.detections)


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


class ExportFormat(str, Enum):
    """Available export formats for model inference.

    Values:
        - ONNX: ONNX format
        - TORCHSCRIPT: TorchScript format

    """

    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"


class RuntimeType(str, Enum):
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

    def to_export_format(self) -> ExportFormat:
        if self == RuntimeType.TORCHSCRIPT_32:
            return ExportFormat.TORCHSCRIPT
        else:
            return ExportFormat.ONNX


class ModelExtension(str, Enum):
    """Supported model extension.

    Values:
        - ONNX: ONNX format
        - TORCHSCRIPT: TorchScript format
        - WEIGHTS: Weights format
    """

    ONNX = "onnx"
    TORCHSCRIPT = "pt"
    WEIGHTS = "pth"

    @classmethod
    def from_runtime_type(cls, runtime_type: RuntimeType):
        if runtime_type in [
            RuntimeType.ONNX_CUDA32,
            RuntimeType.ONNX_TRT32,
            RuntimeType.ONNX_TRT16,
            RuntimeType.ONNX_CPU,
            RuntimeType.ONNX_COREML,
        ]:
            return cls.ONNX
        elif runtime_type == RuntimeType.TORCHSCRIPT_32:
            return cls.TORCHSCRIPT
        else:
            raise ValueError(f"Invalid runtime type: {runtime_type}")


class GPUDevice(PydanticBase):
    """Information about a GPU device."""

    gpu_id: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_used_percentage: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_load_percentage: Optional[float] = None


class GPUInfo(PydanticBase):
    """Information about a GPU driver."""

    gpu_count: Optional[int] = None
    gpu_driver: Optional[str] = None
    gpu_cuda_version: Optional[str] = None
    total_gpu_memory_gb: Optional[float] = None
    devices: Optional[list[GPUDevice]] = None


class SystemInfo(PydanticBase):
    """System information including hardware and software details."""

    focoos_host: Optional[str] = None
    focoos_version: Optional[str] = None
    python_version: Optional[str] = None
    system: Optional[str] = None
    system_name: Optional[str] = None
    cpu_type: Optional[str] = None
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    memory_used_percentage: Optional[float] = None
    available_onnx_providers: Optional[list[str]] = None
    disk_space_total_gb: Optional[float] = None
    disk_space_used_percentage: Optional[float] = None
    pytorch_info: Optional[str] = None
    gpu_info: Optional[GPUInfo] = None
    packages_versions: Optional[dict[str, str]] = None
    environment: Optional[dict[str, str]] = None

    def pprint(self, level: Literal["INFO", "DEBUG"] = "DEBUG"):
        """Pretty print the system info."""
        from focoos.utils.logger import get_logger

        logger = get_logger("SystemInfo", level=level)

        output_lines = ["\n================ 🔍 SYSTEM INFO 🔍 ===================="]
        model_data = self.model_dump()

        if "focoos_host" in model_data and "focoos_version" in model_data:
            output_lines.append(f"focoos: {model_data.get('focoos_host')} (v{model_data.get('focoos_version')})")
            model_data.pop("focoos_host", None)
            model_data.pop("focoos_version", None)

        if "system" in model_data and "system_name" in model_data:
            output_lines.append(f"system: {model_data.get('system')} ({model_data.get('system_name')})")
            model_data.pop("system", None)
            model_data.pop("system_name", None)

        if "cpu_type" in model_data and "cpu_cores" in model_data:
            output_lines.append(f"cpu: {model_data.get('cpu_type')} ({model_data.get('cpu_cores')} cores)")
            model_data.pop("cpu_type", None)
            model_data.pop("cpu_cores", None)

        if "memory_gb" in model_data and "memory_used_percentage" in model_data:
            output_lines.append(
                f"memory_gb: {model_data.get('memory_gb')} ({model_data.get('memory_used_percentage')}% used)"
            )
            model_data.pop("memory_gb", None)
            model_data.pop("memory_used_percentage", None)

        if "disk_space_total_gb" in model_data and "disk_space_used_percentage" in model_data:
            output_lines.append(
                f"disk_space_total_gb: {model_data.get('disk_space_total_gb')} ({model_data.get('disk_space_used_percentage')}% used)"
            )
            model_data.pop("disk_space_total_gb", None)
            model_data.pop("disk_space_used_percentage", None)

        for key, value in model_data.items():
            if key == "gpu_info" and value is not None:
                output_lines.append(f"{key}:")
                output_lines.append(f"  - gpu_count: {value.get('gpu_count')}")
                output_lines.append(f"  - total_memory_gb: {value.get('total_gpu_memory_gb')} GB")
                output_lines.append(f"  - gpu_driver: {value.get('gpu_driver')}")
                output_lines.append(f"  - gpu_cuda_version: {value.get('gpu_cuda_version')}")
                if value.get("devices"):
                    output_lines.append("  - devices:")
                    for device in value.get("devices", []):
                        gpu_memory_used = (
                            f"{device.get('gpu_memory_used_percentage')}%"
                            if device.get("gpu_memory_used_percentage") is not None
                            else "N/A"
                        )
                        gpu_load = (
                            f"{device.get('gpu_load_percentage')}%"
                            if device.get("gpu_load_percentage") is not None
                            else "N/A"
                        )
                        gpu_memory_total = (
                            f"{device.get('gpu_memory_total_gb')} GB"
                            if device.get("gpu_memory_total_gb") is not None
                            else "N/A"
                        )

                        output_lines.append(
                            f"    - GPU {device.get('gpu_id')}: {device.get('gpu_name')}, Memory: {gpu_memory_total} ({gpu_memory_used} used), Load: {gpu_load}"
                        )
            elif isinstance(value, list):
                output_lines.append(f"{key}: {value}")
            elif isinstance(value, dict) and key == "packages_versions":  # Special formatting for packages_versions
                output_lines.append(f"{key}:")
                for pkg_name, pkg_version in value.items():
                    output_lines.append(f"  - {pkg_name}: {pkg_version}")
            elif isinstance(value, dict) and key == "environment":  # Special formatting for environment
                output_lines.append(f"{key}:")
                for env_key, env_value in value.items():
                    output_lines.append(f"  - {env_key}: {env_value}")
            else:
                output_lines.append(f"{key}: {value}")
        output_lines.append("================================================")

        logger.info("\n".join(output_lines))


class ApiKey(PydanticBase):
    """API key for authentication."""

    key: str  # type: ignore


class Quotas(PydanticBase):
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


class User(PydanticBase):
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


@dataclass
class Metrics:
    """
    Collection of training and inference metrics.
    """

    infer_metrics: list[dict] = field(default_factory=list)
    valid_metrics: list[dict] = field(default_factory=list)
    train_metrics: list[dict] = field(default_factory=list)
    iterations: Optional[int] = None
    best_valid_metric: Optional[dict] = None


class ModelFamily(str, Enum):
    """Enumerazione delle famiglie di modelli disponibili"""

    DETR = "fai_detr"
    MASKFORMER = "fai_mf"
    BISENETFORMER = "bisenetformer"
    IMAGE_CLASSIFIER = "fai_cls"


# This should not be a dataclass, but their child must be
class DictClass(OrderedDict):
    def to_tuple(self) -> tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __post_init__(self):
        """Check the BasicContainer dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        for _field in class_fields:
            v = getattr(self, _field.name)
            if v is not None:
                self[_field.name] = v

    def __reduce__(self):
        state_dict = {field.name: getattr(self, field.name) for field in fields(self)}
        return (self.__class__.__new__, (self.__class__,), state_dict)


@dataclass
class ModelConfig(DictClass):
    num_classes: int


@dataclass
class ModelOutput(DictClass):
    """Model output base container."""

    loss: Optional[dict]


@dataclass
class DatasetEntry(DictClass):
    image: Optional[Tensor] = None
    height: Optional[int] = None
    width: Optional[int] = None
    instances: Optional[Instances] = None
    file_name: Optional[str] = None
    image_id: Optional[int] = None


class DatasetSplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def get_gpus_count():
    try:
        import torch.cuda

        return torch.cuda.device_count()
    except ImportError:
        return 0


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
    output_dir: str = MODELS_DIR
    ckpt_dir: Optional[str] = None
    init_checkpoint: Optional[str] = None
    resume: bool = False
    # Logistics params
    num_gpus: int = get_gpus_count()
    device: str = "cuda"
    workers: int = 4
    amp_enabled: bool = True
    ddp_broadcast_buffers: bool = False
    ddp_find_unused: bool = True
    checkpointer_period: int = 1000
    checkpointer_max_to_keep: int = 1
    eval_period: int = 50
    log_period: int = 20
    samples: int = 9
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
    scheduler: Literal["POLY", "FIXED", "COSINE", "MULTISTEP"] = "MULTISTEP"
    scheduler_extra: Optional[dict] = None
    optimizer: Literal["ADAMW", "SGD", "RMSPROP"] = "ADAMW"
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

    # Sync to hub
    sync_to_hub: bool = False


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
    def classes(self) -> List[str]:  #!TODO: check if this is correct
        if self.task == Task.DETECTION or self.task == Task.INSTANCE_SEGMENTATION:
            assert self.thing_classes is not None, "thing_classes is required for detection and instance segmentation"
            return self.thing_classes
        if self.task == Task.SEMSEG:
            # fixme: not sure for panoptic
            assert self.stuff_classes is not None, "stuff_classes is required for semantic segmentation"
            return self.stuff_classes
        if self.task == Task.CLASSIFICATION:
            assert self.thing_classes is not None, "thing_classes is required for classification"
            return self.thing_classes
        raise ValueError(f"Task {self.task} not supported")

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
class DetectronDict:
    file_name: str
    height: Optional[int] = None
    width: Optional[int] = None
    image_id: Optional[Union[str, int]] = None
    sem_seg_file_name: Optional[str] = None
    pan_seg_file_name: Optional[str] = None
    annotations: Optional[list[dict]] = None
    segments_info: Optional[list[dict]] = None


@dataclass
class ModelInfo(DictClass):
    """
    Comprehensive metadata for a Focoos model.

    This dataclass encapsulates all relevant information required to identify, configure, and evaluate a model
    within the Focoos platform. It is used for serialization, deserialization, and programmatic access to model
    properties.

    Attributes:
        name (str): Human-readable name or unique identifier for the model.
        model_family (ModelFamily): The model's architecture family (e.g., RTDETR, M2F).
        classes (list[str]): List of class names that the model can detect or segment.
        im_size (int): Input image size (usually square, e.g., 640).
        task (Task): Computer vision task performed by the model (e.g., detection, segmentation).
        config (dict): Model-specific configuration parameters.
        ref (Optional[str]): Optional unique reference string for the model.
        focoos_model (Optional[str]): Optional Focoos base model identifier.
        status (Optional[ModelStatus]): Current status of the model (e.g., training, ready).
        description (Optional[str]): Optional human-readable description of the model.
        train_args (Optional[TrainerArgs]): Optional training arguments used to train the model.
        weights_uri (Optional[str]): Optional URI or path to the model weights.
        val_dataset (Optional[str]): Optional name or reference of the validation dataset.
        val_metrics (Optional[dict]): Optional dictionary of validation metrics (e.g., mAP, accuracy).
        focoos_version (Optional[str]): Optional Focoos version string.
        latency (Optional[list[LatencyMetrics]]): Optional list of latency measurements for different runtimes.
        updated_at (Optional[str]): Optional ISO timestamp of the last update.
    """

    name: str
    model_family: ModelFamily
    classes: list[str]
    im_size: int
    task: Task
    config: dict
    ref: Optional[str] = None
    focoos_model: Optional[str] = None
    status: Optional[ModelStatus] = None
    description: Optional[str] = None
    train_args: Optional[TrainerArgs] = None
    weights_uri: Optional[str] = None
    val_dataset: Optional[str] = None
    val_metrics: Optional[dict] = None  # TODO: Consider making metrics explicit in the future
    focoos_version: Optional[str] = None
    latency: Optional[list[LatencyMetrics]] = None
    training_info: Optional[TrainingInfo] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_json(cls, path: str):
        """
        Load ModelInfo from a JSON file.

        Args:
            path (str): Path to the JSON file containing model metadata.

        Returns:
            ModelInfo: An instance of ModelInfo populated with data from the file.
        """
        with open(path, encoding="utf-8") as f:
            model_info_json = json.load(f)
        model_info = cls(
            name=model_info_json["name"],
            ref=model_info_json.get("ref", None),
            model_family=ModelFamily(model_info_json["model_family"]),
            classes=model_info_json["classes"],
            im_size=int(model_info_json["im_size"]),
            status=ModelStatus(model_info_json.get("status")) if model_info_json.get("status") else None,
            task=Task(model_info_json["task"]),
            focoos_model=model_info_json.get("focoos_model", None),
            config=model_info_json["config"],
            description=model_info_json.get("description", None),
            train_args=TrainerArgs(**model_info_json["train_args"])
            if "train_args" in model_info_json and model_info_json["train_args"] is not None
            else None,
            weights_uri=model_info_json.get("weights_uri", None),
            val_dataset=model_info_json.get("val_dataset", None),
            latency=[LatencyMetrics(**latency) for latency in model_info_json.get("latency", [])]
            if "latency" in model_info_json and model_info_json["latency"] is not None
            else None,
            updated_at=model_info_json.get("updated_at", None),
            focoos_version=model_info_json.get("focoos_version", None),
            val_metrics=model_info_json.get("val_metrics", None),
            training_info=TrainingInfo(**model_info_json["training_info"])
            if "training_info" in model_info_json and model_info_json["training_info"] is not None
            else None,
        )
        return model_info

    def dump_json(self, path: str):
        """
        Serialize ModelInfo to a JSON file.

        Args:
            path (str): Path where the JSON file will be saved.
        """
        data = asdict(self)
        # Note: config_class is not included; if needed, convert to string here.

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def pprint(self):
        """
        Pretty-print the main model information using the Focoos logger.
        """
        from focoos.utils.logger import get_logger

        logger = get_logger("model_info")
        logger.info(
            f"""
            📋 Name: {self.name}
            📝 Description: {self.description}
            👪 Family: {self.model_family}
            🔗 Focoos Model: {self.focoos_model}
            🎯 Task: {self.task}
            🏷️ Classes: {self.classes}
            🖼️ Im size: {self.im_size}
            """
        )


@dataclass
class ExportCfg:
    """Configuration for model export.

    Args:
        out_dir: Output directory for exported model
        onnx_opset: ONNX opset version to use
        onnx_dynamic: Whether to use dynamic axes in ONNX export
        onnx_simplify: Whether to simplify ONNX model
        model_fuse: Whether to fuse model layers
        format: Export format ("onnx" or "torchscript")
        device: Device to use for export
    """

    out_dir: str
    onnx_opset: int = 17
    onnx_dynamic: bool = True
    onnx_simplify: bool = True
    model_fuse: bool = True
    format: Literal["onnx", "torchscript"] = "onnx"
    device: Optional[str] = "cuda"


@dataclass
class DynamicAxes:
    """Dynamic axes for model export."""

    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict


class ArtifactName(str, Enum):
    """Model artifact type."""

    WEIGHTS = "model_final.pth"
    ONNX = "model.onnx"
    PT = "model.pt"
    INFO = "model_info.json"
    METRICS = "metrics.json"
    LOGS = "log.txt"
