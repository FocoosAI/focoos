# 🚀 Focoos HUB

The FocoosHUB class is your main interface for interacting with the Focoos cloud platform. It provides comprehensive functionality for managing models, datasets, and performing cloud operations.

## Getting Started

### Authentication

Before using the HUB, you need to authenticate with your API key:

```python
from focoos import FocoosHUB

# Option 1: Use API key from configuration
hub = FocoosHUB()

# Option 2: Explicitly provide API key
hub = FocoosHUB(api_key="your-api-key-here")
```

### Get User Information

Check your account details and quotas:

```python
user_info = hub.get_user_info()

print(f"Email: {user_info.email}")
print(f"Company: {user_info.company}")
print(f"Storage used: {user_info.quotas.used_storage_gb}GB")
print(f"Inferences used: {user_info.quotas.total_inferences}")
```

## Working with Models

### List Your Models

Get an overview of all models in your account:

```python
models = hub.list_remote_models()

for model in models:
    print(f"Model: {model.name}")
    print(f"  - Reference: {model.ref}")
    print(f"  - Task: {model.task}")
    print(f"  - Status: {model.status}")
    print(f"  - Created: {model.created_at}")
```

### Get Model Information

Retrieve detailed information about a specific model:

```python
model_ref = "your-model-reference"
model_info = hub.get_model_info(model_ref)

print(f"Model Name: {model_info.name}")
print(f"Description: {model_info.description}")
print(f"Task: {model_info.task}")
print(f"Classes: {model_info.classes}")
print(f"Image Size: {model_info.im_size}")
print(f"Status: {model_info.status}")

# Access training information if available
if model_info.training:
    print(f"Training Status: {model_info.training.status}")
    print(f"Training Progress: {model_info.training.progress}%")
```

### Get Remote Model Instance

Get a remote model instance for cloud-based operations:

```python
model_ref = "your-model-reference"
remote_model = hub.get_remote_model(model_ref)

# This model can be used for remote inference
results = remote_model.infer("path/to/image.jpg", threshold=0.5)

# Get model training information
training_info = remote_model.train_info()
print(f"Training status: {training_info.status}")

# Get model metrics
metrics = remote_model.metrics()
print(f"Validation mAP: {metrics.map}")
```

## Working with Datasets

### List Available Datasets

View datasets you own and optionally shared datasets:

```python
# List only your datasets
my_datasets = hub.list_remote_datasets()

# List your datasets and shared datasets
all_datasets = hub.list_remote_datasets(include_shared=True)

for dataset in all_datasets:
    print(f"Dataset: {dataset.name}")
    print(f"  - Reference: {dataset.ref}")
    print(f"  - Task: {dataset.task}")
    print(f"  - Layout: {dataset.layout}")
    if dataset.spec:
        print(f"  - spec.train_length: {dataset.spec.train_length}")
        print(f"  - spec.valid_length: {dataset.spec.valid_length}")
        print(f"  - spec.size_mb: {dataset.spec.size_mb}")
```

### Working with Remote Datasets

Get a remote dataset instance and work with it:

```python
dataset_ref = "your-dataset-reference"
remote_dataset = hub.get_remote_dataset(dataset_ref)

print(f"Dataset Name: {remote_dataset.name}")
print(f"Task: {remote_dataset.task}")
print(f"Layout: {remote_dataset.layout}")

# Download dataset data
local_path = remote_dataset.download_data("./datasets")
print(f"Dataset downloaded to: {local_path}")
```


## Error Handling

The HUB client raises `ValueError` exceptions for API errors:

```python
try:
    model_info = hub.get_model_info("non-existent-model")
except ValueError as e:
    print(f"Error retrieving model: {e}")

try:
    models = hub.list_remote_models()
except ValueError as e:
    print(f"Error listing models: {e}")
```

## Configuration

The HUB client uses configuration from the global `FOCOOS_CONFIG`:

```python
from focoos.config import FOCOOS_CONFIG

# Check current configuration
print(f"API Key: {FOCOOS_CONFIG.focoos_api_key}")
print(f"Log Level: {FOCOOS_CONFIG.focoos_log_level}")
print(f"Runtime type: {FOCOOS_CONFIG.runtime_type}")
print(f"Warmup iter: {FOCOOS_CONFIG.warmup_iter}")

# The HUB client will use these values by default
hub = FocoosHUB()  # Uses FOCOOS_CONFIG values
```

## Advanced Usage

### Model Training Integration

When training models locally, you can sync them to the HUB:

```python
from focoos.models.focoos_model import FocoosModel
from focoos.ports import TrainerArgs
from focoos import ModelManager

# Load a model for training
model = ModelManager.get("fai-detr-l-obj365")

# Configure training with HUB sync
train_args = TrainerArgs(
    max_iters=1000,
    batch_size=16,
    sync_to_hub=True  # This will automatically create a remote model
)

# Train the model (this will sync to HUB)
model.train(train_args, train_dataset, val_dataset, hub=hub)
```

### Monitoring Training

Monitor training progress of remote models:

```python
remote_model = hub.get_remote_model("training-model-ref")

# Get training info
training_info = remote_model.train_info()

if training_info:
    print(f"Algorithm Name: {training_info.algorithm_name}")
    print(f"Instance Device: {training_info.instance_device}")
    print(f"Instance Type: {training_info.instance_type}")
    print(f"Volume Size: {training_info.volume_size}")
    print(f"Main Status: {training_info.main_status}")
    print(f"Failure Reason: {training_info.failure_reason}")
    print(f"Status Transitions: {training_info.status_transitions}")
    print(f"Start Time: {training_info.start_time}")
    print(f"End Time: {training_info.end_time}")
    print(f"Artifact Location: {training_info.artifact_location}")

# Get training logs
logs = remote_model.train_logs()
for log_entry in logs:
    print(log_entry)
```

## See Also

- [Remote Inference](remote_inference.md) - Learn about cloud-based inference
- [Overview](overview.md) - Understand the HUB architecture
- [API Reference](../api/hub.md) - Detailed API documentation
