# Main Concepts

## FocoosModel

The `FocoosModel` class is the main interface for working with computer vision models in Focoos. It provides high-level methods for training, testing, inference, and model export while handling preprocessing and postprocessing automatically.

### Key Features

- **End-to-End Inference**: Automatic preprocessing and postprocessing
- **Training Support**: Built-in training pipeline with distributed training support
- **Model Export**: Export to ONNX and TorchScript formats
- **Performance Benchmarking**: Built-in latency and throughput measurement
- **Hub Integration**: Seamless integration with Focoos Hub for model sharing
- **Multiple Input Formats**: Support for PIL Images, NumPy arrays, and PyTorch tensors

### Loading Strategies

The primary method for loading models is using the `ModelManager.get()` (see [`ModelManager`](/focoos/api/model_manager/#focoos.model_manager.ModelManager)). It supports multiple loading strategies based on the input parameters. The return value is a [Focoos Model](#focoosmodel).

The ModelManager employs different loading strategies based on the input:

#### 1. From Focoos Hub

The Focoos Hub is a cloud-based model repository where you can store, share, and collaborate on models. This method enables seamless model downloading and caching from the hub using the `hub://` protocol.

**When to use**: Load models shared by other users, access your own cloud-stored models, or work with models that require authentication.

**Requirements**: Valid API key for private models, internet connection for initial download.

```python
from focoos import FocoosHub, ModelManager
# Loading from hub using hub:// protocol
# The model is automatically downloaded and cached locally
hub = FocoosHUB(api_key="your_api_key")
model = ModelManager.get("hub://model_reference", hub=hub)

# Loading with custom configuration override
model = ModelManager.get(
    "hub://model_reference",
    hub=hub,
    cache=True,  # Cache for faster subsequent loads
    config_parameter=your_value # Override single config parameter
)
```

#### 2. From Model Registry

The Model Registry contains curated, pretrained models that are immediately available without download. These models are optimized, tested, and ready for production use across various computer vision tasks.

**When to use**: Start with proven, high-quality pretrained models, baseline experiments, or when you need reliable performance without customization.

**Requirements**: No internet connection needed, models are bundled with the library.

```python
from focoos import ModelRegistry, ModelManager
# Loading pretrained models from registry
# Object detection model trained on COCO dataset
model = ModelManager.get("fai-detr-l-coco")

# Semantic segmentation model for ADE20K dataset
model = ModelManager.get("fai-mf-l-ade")

# Check available models first
available_models = ModelRegistry.list_models()
print("Available models:", available_models)

# Get detailed information before loading
model_info = ModelRegistry.get_model_info("fai-detr-l-coco")
print(f"Classes: {len(model_info.classes)}, Task: {model_info.task}")
```

**Available Model Categories**:

 - **Object Detection**: `fai-detr-l-coco`, `fai-detr-m-coco`, `fai-detr-l-obj365`
 - **Instance Segmentation**: `fai-mf-l-coco-ins`, `fai-mf-m-coco-ins`, `fai-mf-s-coco-ins`
 - **Semantic Segmentation**: `fai-mf-l-ade`, `fai-mf-m-ade`, `bisenetformer-l-ade`, `bisenetformer-m-ade`, `bisenetformer-s-ade`

#### 3. From Local Directory

Load models from your local filesystem, whether they're custom-trained models or models stored in non-standard locations. This method provides maximum flexibility for local development and deployment scenarios.

**When to use**: Load custom-trained models, work with locally stored models, integrate with existing model storage systems, or work in offline environments.

**Requirements**: Valid model directory containing model artifacts (weights, configuration, metadata).

```python
# Loading with custom models directory
model = ModelManager.get("/custom/models/dir/my_model")

# Expected directory structure:
# /path/to/local/model/
# ├── model_info.json     # Model metadata and configuration
# ├── model_final.pth     # Model weights (optional)

# Loading with configuration override
model = ModelManager.get(
    "/custom/models/dir/local_model",
    arg1=value1,
    arg2=value2,
)
```

#### 4. From ModelInfo Object

The [`ModelInfo`](/focoos/api/ports/#focoos.ports.ModelInfo) class represents comprehensive model metadata including architecture specifications, training configuration, class definitions, and performance metrics. This method provides the most programmatic control over model instantiation.

**When to use**: Programmatically construct models, work with dynamic configurations, integrate with custom model management systems, or when you need fine-grained control over model instantiation.

**Requirements**: Properly constructed ModelInfo object with valid configuration parameters.

```python
from focoos import ModelInfo, ModelFamily, Task
# Loading from JSON file
model_info = ModelInfo.from_json("path/to/model_info.json")
model = ModelManager.get("any_name", model_info=model_info)

# Programmatically creating ModelInfo

model_info = ModelInfo(
    name="custom_detector",
    model_family=ModelFamily.DETR,
    classes=["person", "car", "bicycle"],
    im_size=640,  # Square image (640x640), or use (640, 480) for non-square (height, width)
    task=Task.DETECTION,
    config={
        "num_classes": 3,
        "backbone_config": {"depth": 50, "model_type": "resnet"},
        "threshold": 0.5
    },
    weights_uri="path/to/weights.pth",  # Optional
    description="Custom object detector"
)

model = ModelManager.get("custom_detector", model_info=model_info)
```

### Inference

Performs end-to-end inference on input images with automatic preprocessing and postprocessing. The model accepts input images in various formats including:

- PIL Image objects (`PIL.Image.Image`)
- NumPy arrays (`numpy.ndarray`)
- PyTorch tensors (`torch.Tensor`)

The input images are automatically preprocessed to the correct size and format required by the model. After inference, the raw model outputs are postprocessed into a standardized [`FocoosDetections`](/focoos/api/ports/#focoos.ports.FocoosDetections) format that provides easy access to:

- Detected object classes and confidence scores
- Bounding box coordinates
- Segmentation masks (for segmentation models)
- Additional model-specific outputs

This provides a simple, unified interface for running inference regardless of the underlying model architecture or task.

**Parameters:**
- `image`: Input image in various supported formats (`PIL.Image.Image`, `numpy.ndarray`, `torch.Tensor`, local or remote path)
- `threshold`: detections threshold
- `annotate`: if you want to annotate detections on provided image
- `**kwargs`: Additional arguments passed to postprocessing

**Returns:** [`FocoosDetections`](/focoos/api/ports/#focoos.ports.FocoosDetections) containing detection/segmentation results

**Example:**
```python
from PIL import Image

# Load an image
im_path = "example.jpg"

# Run inference
detections = model.infer(im_path,threshold=0.5,annotate=True)

# Access results
for detection in detections.detections:
    print(f"Class: {detection.label}, Confidence: {detection.conf}")
    print(f"Bounding box: {detection.bbox}")

Image.fromarray(detections.image)
```

### Training
Trains the model on provided datasets. The training function accepts:

- `args`: Training configuration ([TrainerArgs](/focoos/api/ports/#focoos.ports.TrainerArgs)) specifying the main hyperparameters, among which:
  - `run_name`: Name for the training run
  - `output_dir`: Name for the output folder
  - `num_gpus`: Number of GPUs to use (must be >= 1)
  - `sync_to_hub`: For tracking the experiment on the Focoos Hub.
  -`batch_size`, `learning_rate`, `max_iters` and other hyperparameters
- `data_train`: Training dataset (MapDataset)
- `data_val`: Validation dataset (MapDataset)
- `hub`: Optional FocoosHUB instance for experiment tracking

The data can be obtained using the [AutoDataset](/focoos/api/auto_dataset/#focoos.data.auto_dataset.AutoDataset) helper.

After the training is complete, the model will have updated weights and can be used for inference or export. Furthermore, in the `output_dir` can be found the model metadata (`model_info.json`) and the PyTorch weights (`model_final.pth`).

**Example:**
```python
from focoos import TrainerArgs
from focoos.data import MapDataset

# Configure training
train_args = TrainerArgs(
    run_name="my_custom_model",
    max_iters=5000,
    batch_size=16,
    learning_rate=1e-4,
    num_gpus=2,
    sync_to_hub=True,
)

# Train the model
model.train(train_args, train_dataset, val_dataset, hub=hub)
```

[Here](training.md) you can find an extensive training tutorial.

### Model Export

Exports the model to different runtime formats for optimized inference. The main function arguments are:
 - `runtime_type`: specify the target runtime and must be one of the supported (see [RuntimeType](/focoos/api/ports/#focoos.ports.RuntimeType))
 - `out_dir`: the destination folder for the exported model
 - `image_size`: the target image size, as an optional integer (square) or tuple (height, width) for non-square images

The function returns an [`InferModel`](#infer-model) instance for the exported model.

**Example:**
```python
# Export to ONNX with TensorRT optimization
infer_model = model.export(
    runtime_type=RuntimeType.ONNX_TRT16,
    out_dir="./exported_models",
    overwrite=True
)

# Use exported model for fast inference
fast_detections = infer_model(image)
```
---

## Infer Model
The `InferModel` class represents an optimized model for inference, typically created through the export process of a `FocoosModel`. It provides a streamlined interface focused on fast and efficient inference while maintaining the same input/output format as the original model.

### Key Features

- **Optimized Performance**: Models are optimized for the target runtime (e.g., TensorRT, ONNX)
- **Consistent Interface**: Uses the same input/output format as FocoosModel
- **Resource Management**: Proper cleanup of runtime resources when no longer needed
- **Multiple Input Formats**: Support for PIL Images, NumPy arrays, and PyTorch tensors

### Initialization

InferModel instances are typically created through the `export()` method of a [FocoosModel](#focoosmodel), which handles the model optimization and conversion process. This method allows you to specify the target runtime (see the availables in [`Runtimetypes`](/focoos/api/ports/#focoos.ports.RuntimeType)) and the output directory for the exported model. The `export()` method returns an `InferModel` instance that is optimized for fast and efficient inference.

**Example:**
```python
# Export the model to ONNX format
infer_model = model.export(
    runtime_type=RuntimeType.TORCHSCRIPT_32,
    out_dir="./exported_models"
)

# Use the exported model for inference
results = infer_model(input_image)
```

### Inference

Performs end-to-end inference on input images with automatic preprocessing and postprocessing on the selected runtime. The model accepts input images in various formats including:

- PIL Image objects (`PIL.Image.Image`)
- NumPy arrays (`numpy.ndarray`)
- PyTorch tensors (`torch.Tensor`)

The input images are automatically preprocessed to the correct size and format required by the model. After inference, the raw model outputs are postprocessed into a standardized [`FocoosDetections`](./api/ports/#focoos.ports.FocoosDetections) format that provides easy access to:

- Detected object classes and confidence scores
- Bounding box coordinates
- Segmentation masks (for segmentation models)
- Additional model-specific outputs

This provides a simple, unified interface for running inference regardless of the underlying model architecture or task.

**Parameters:**
- `image`: Input image in various supported formats (`PIL.Image.Image`, `numpy.ndarray`, `torch.Tensor`, local or remote path)
- `threshold`: detections threshold
- `annotate`: if you want to annotate detections on provided image
- `**kwargs`: Additional arguments passed to postprocessing

**Returns:** [`FocoosDetections`](/focoos/api/ports/#focoos.ports.FocoosDetections) containing detection/segmentation results

**Example:**
```python
from PIL import Image

# Load an image
image_path = "example.jpg"

# Run inference
infer_model = model.export(
    runtime_type=RuntimeType.TORCHSCRIPT_32,
    out_dir="./exported_models"
)
detections = infer_model.infer(image_path,threshold=0.5, annotate = True)

# Access results
for detection in detections.detections:
    print(f"Class: {detection.label}, Confidence: {detection.conf}")
    print(f"Bounding box: {detection.bbox}")
```
