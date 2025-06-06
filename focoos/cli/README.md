# Focoos CLI

A modern command-line interface for the Focoos computer vision framework, providing streamlined commands for training, validation, inference, and model management.

## Installation

```bash
pip install focoos
```

## Usage

The Focoos CLI uses a clean, modern command syntax:

```bash
focoos COMMAND [OPTIONS]
```

Where:
- **COMMAND**: Main commands like `train`, `val`, `predict`, `export`, `benchmark`
- **OPTIONS**: Command-specific flags and parameters

## Available Commands

### Main Commands
- **`train`** - Train a model on a dataset
- **`val`** - Validate a model on a dataset
- **`predict`** - Run inference on images/videos
- **`export`** - Export a model to different formats
- **`benchmark`** - Benchmark model performance

### Utility Commands
- **`version`** - Show Focoos version information
- **`checks`** - Run system checks and display system information
- **`settings`** - Show current Focoos configuration settings

## Examples

### Training

```bash
# Train a detection model
focoos train --model fai-detr-m-coco --dataset coco8.yaml --im-size 640

# Train with custom settings
focoos train --model fai-detr-m-coco --dataset my_dataset --max-iters 5000 --batch-size 8 --learning-rate 0.001

# Train with early stopping
focoos train --model fai-detr-m-coco --dataset coco8.yaml --early-stop --patience 20
```

### Validation

```bash
# Validate a model
focoos val --model fai-detr-m-coco --dataset coco8.yaml --im-size 640

# Validate with custom batch size
focoos val --model fai-detr-m-coco --dataset my_dataset --batch-size 32

# Validate with specific checkpoint
focoos val --model fai-detr-m-coco --dataset coco8.yaml --init-checkpoint path/to/checkpoint.pth
```

### Prediction

```bash
# Predict on a single image
focoos predict --model fai-detr-m-coco --source image.jpg

# Predict with custom confidence threshold
focoos predict --model fai-detr-m-coco --source images/ --conf 0.5

# Predict from URL and save results
focoos predict --model fai-detr-m-coco --source https://example.com/image.jpg --save --save-json

# Predict with custom output directory
focoos predict --model fai-detr-m-coco --source video.mp4 --output-dir ./results
```

### Export

```bash
# Export to ONNX (default)
focoos export --model fai-detr-m-coco --im-size 640

# Export to TorchScript
focoos export --model fai-detr-m-coco --format torchscript --device cuda

# Export with custom output directory
focoos export --model fai-detr-m-coco --format onnx --output-dir ./exports --overwrite
```

### Benchmark

```bash
# Benchmark model performance
focoos benchmark --model fai-detr-m-coco --iterations 100 --device cuda

# Benchmark with specific image size
focoos benchmark --model fai-detr-m-coco --im-size 1024 --iterations 50

# Benchmark on CPU
focoos benchmark --model fai-detr-m-coco --device cpu --iterations 20
```

## Utility Commands

```bash
# Show version information
focoos version

# Run system checks and show system info
focoos checks

# Show current configuration settings
focoos settings

# Get help for the main CLI
focoos --help

# Get help for any specific command
focoos train --help
focoos predict --help
focoos export --help
```

## Common Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--model` | Model name or path | Required | `--model fai-detr-m-coco` |
| `--dataset` | Dataset name | Required for train/val | `--dataset coco8.yaml` |
| `--source` | Image/video path or URL | Required for predict | `--source image.jpg` |
| `--im-size` | Image size | 640 | `--im-size 1024` |
| `--batch-size` | Batch size | 16 | `--batch-size 32` |
| `--max-iters` | Training iterations | 3000 | `--max-iters 5000` |
| `--learning-rate` | Learning rate | 5e-4 | `--learning-rate 0.001` |
| `--conf` | Confidence threshold | 0.25 | `--conf 0.5` |
| `--save` | Save results | true | `--save` |
| `--workers` | Number of workers | 4 | `--workers 8` |
| `--output-dir` | Output directory | varies | `--output-dir ./results` |
| `--device` | Device (cuda/cpu) | cuda | `--device cpu` |

## Training Options

### Basic Training
```bash
focoos train --model fai-detr-m-coco --dataset coco8.yaml
```

### Advanced Training Options
```bash
focoos train \
  --model fai-detr-m-coco \
  --dataset coco8.yaml \
  --im-size 640 \
  --batch-size 16 \
  --max-iters 5000 \
  --learning-rate 5e-4 \
  --weight-decay 0.02 \
  --scheduler MULTISTEP \
  --optimizer ADAMW \
  --early-stop \
  --patience 10 \
  --ema-enabled \
  --amp-enabled
```

### Scheduler Options
- `MULTISTEP` (default)
- `COSINE`
- `POLY`
- `FIXED`

### Optimizer Options
- `ADAMW` (default)
- `SGD`
- `RMSPROP`

## Runtime Types

For prediction and benchmarking:
- `onnx_cuda32`: ONNX with CUDA FP32 (default)
- `onnx_cpu`: ONNX on CPU
- `torchscript_32`: TorchScript FP32

## Export Formats

- `onnx`: ONNX format (default)
- `torchscript`: TorchScript format

## Dataset Layouts

- `roboflow_coco`: Roboflow COCO format (default)
- `yolo`: YOLO format
- `coco`: COCO format

## Error Handling

The CLI provides clear error messages and validation:

```bash
# Missing required argument
$ focoos train --dataset coco8.yaml
Error: Missing option '--model'

# Invalid parameter value
$ focoos train --model fai-detr-m-coco --dataset coco8.yaml --scheduler INVALID
Error: Scheduler must be one of ['POLY', 'FIXED', 'COSINE', 'MULTISTEP']

# Get help for specific command
$ focoos train --help
# Shows all available options with descriptions
```

## Integration with Python

You can also access CLI functionality programmatically:

```python
from focoos.cli.commands import train_command, predict_command

# Train a model programmatically
train_command(
    model_name="fai-detr-m-coco",
    dataset_name="coco8.yaml",
    im_size=640,
    max_iters=3000
)

# Run prediction programmatically
predict_command(
    model_name="fai-detr-m-coco",
    source="image.jpg",
    conf=0.5
)
```

## Configuration

Configure Focoos settings through environment variables or configuration files. Check current settings with:

```bash
focoos settings
```

Common environment variables:
- `FOCOOS_RUNTIME_TYPE`: Default runtime type
- `FOCOOS_LOG_LEVEL`: Logging level
- `FOCOOS_API_KEY`: API key for Focoos services

## Features

✅ **Modern CLI**: Clean, intuitive command structure with rich help
✅ **Type Safety**: Built-in validation for arguments and options
✅ **Comprehensive**: Complete coverage of Focoos functionality
✅ **Auto-completion**: Tab completion support (when shell completion is configured)
✅ **Rich Output**: Detailed progress information and results
✅ **Flexible**: Extensive customization options for all commands
✅ **Error Handling**: Clear error messages with helpful suggestions
