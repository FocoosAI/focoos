# Focoos CLI

A modern, comprehensive command-line interface for the Focoos computer vision framework. The CLI provides streamlined commands for training, validation, inference, model management, and hub interactions with rich terminal output and extensive configuration options.

## 🚀 Features

- ✅ **Modern CLI Design**: Built with Typer for rich, interactive command-line experience
- ✅ **Comprehensive Coverage**: Complete access to all Focoos functionality
- ✅ **Type Safety**: Built-in validation for arguments and options with helpful error messages
- ✅ **Rich Output**: Colored output, progress indicators, and detailed feedback
- ✅ **Flexible Configuration**: Extensive customization options for all operations
- ✅ **Multi-Platform**: Cross-platform support (Linux, macOS, Windows)
- ✅ **Programmatic API**: Commands can be used both via CLI and Python imports
- ✅ **Hub Integration**: Seamless integration with Focoos Hub for model and dataset management

## 📦 Installation

```bash
pip install focoos
```

## 🎯 Quick Start

```bash
# Check installation
focoos version

# Run system checks
focoos checks

# Train a model
focoos train --model fai-detr-m-coco --dataset mydataset.zip --im-size 640

# Run inference
focoos predict --model fai-detr-m-coco --source image.jpg

# Export model
focoos export --model fai-detr-m-coco --format onnx
```

## 📚 Usage

The Focoos CLI uses a clean, modern command syntax:

```bash
focoos COMMAND [OPTIONS]
```

Where:
- **COMMAND**: Main operations like `train`, `val`, `predict`, `export`, `benchmark`, `hub`
- **OPTIONS**: Command-specific flags and parameters with intelligent defaults

## 🛠️ Available Commands

### Core Commands
| Command | Description | Example |
|---------|-------------|---------|
| **`train`** | Train models on datasets | `focoos train --model fai-detr-m-coco --dataset data.zip` |
| **`val`** | Validate model performance | `focoos val --model fai-detr-m-coco --dataset data.zip` |
| **`predict`** | Run inference on images | `focoos predict --model fai-detr-m-coco --source image.jpg` |
| **`export`** | Export models to different formats | `focoos export --model fai-detr-m-coco --format onnx` |
| **`benchmark`** | Benchmark model performance | `focoos benchmark --model fai-detr-m-coco --iterations 100` |

### Hub Commands
| Command | Description | Example |
|---------|-------------|---------|
| **`hub models`** | List available pretrained models | `focoos hub models` |
| **`hub datasets`** | List available datasets | `focoos hub datasets --include-shared` |

### Utility Commands
| Command | Description | Example |
|---------|-------------|---------|
| **`version`** | Show Focoos version information | `focoos version` |
| **`checks`** | Run system diagnostics | `focoos checks` |
| **`settings`** | Show current configuration | `focoos settings` |

## 📖 Detailed Examples

### 🏋️ Training

```bash
# Basic training
focoos train --model fai-detr-m-coco --dataset mydataset.zip --im-size 640

# Advanced training with custom hyperparameters
focoos train \
  --model fai-detr-m-coco \
  --dataset mydataset.zip \
  --im-size 640 \
  --batch-size 16 \
  --max-iters 5000 \
  --learning-rate 1e-3 \
  --scheduler COSINE \
  --optimizer ADAMW \
  --early-stop \
  --patience 20 \
  --ema-enabled

# Multi-GPU distributed training
focoos train \
  --model fai-detr-m-coco \
  --dataset mydataset.zip \
  --num-gpus 4 \
  --batch-size 32 \
  --workers 8

# Resume training from checkpoint
focoos train \
  --model fai-detr-m-coco \
  --dataset mydataset.zip \
  --resume \
  --init-checkpoint path/to/checkpoint.pth
```

### 🔍 Validation

```bash
# Basic validation
focoos val --model fai-detr-m-coco --dataset mydataset.zip --im-size 640

# Validation with specific checkpoint
focoos val \
  --model fai-detr-m-coco \
  --dataset mydataset.zip \
  --init-checkpoint path/to/best_model.pth \
  --batch-size 32

# Multi-GPU validation
focoos val \
  --model fai-detr-m-coco \
  --dataset mydataset.zip \
  --num-gpus 2 \
  --batch-size 64
```

### 🎯 Prediction & Inference

```bash
# Single image inference
focoos predict --model fai-detr-m-coco --source image.jpg


# URL inference with custom runtime
focoos predict \
  --model fai-detr-m-coco \
  --source https://example.com/image.jpg \
  --runtime onnx_cuda32 \
  --output-dir ./url_results
```

### 📤 Model Export

```bash
# Export to ONNX (default)
focoos export --model fai-detr-m-coco --im-size 640

# Export to TorchScript
focoos export \
  --model fai-detr-m-coco \
  --format torchscript \
  --device cuda \
  --im-size 1024

# Export with custom settings
focoos export \
  --model fai-detr-m-coco \
  --format onnx \
  --output-dir ./exports \
  --onnx-opset 16 \
  --overwrite

# CPU-optimized export
focoos export \
  --model fai-detr-m-coco \
  --format onnx \
  --device cpu \
  --output-dir ./cpu_models
```

### ⚡ Benchmarking

```bash
# Basic GPU benchmark
focoos benchmark --model fai-detr-m-coco --iterations 100 --device cuda

# Comprehensive benchmark with custom settings
focoos benchmark \
  --model fai-detr-m-coco \
  --im-size 1024 \
  --iterations 200 \
  --device cuda

# CPU benchmark
focoos benchmark \
  --model fai-detr-m-coco \
  --device cpu \
  --iterations 50 \
  --im-size 640

# Memory-constrained benchmark
focoos benchmark \
  --model fai-detr-m-coco \
  --im-size 512 \
  --iterations 30
```

### 🌐 Hub Integration

```bash
# List all available models
focoos hub models

# List your private datasets
focoos hub datasets

# List both private and shared datasets
focoos hub datasets --include-shared
```

## ⚙️ Configuration Options

### Common Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Model name or path | **Required** | `fai-detr-m-coco`, `path/to/model` |
| `--dataset` | Dataset name or path | **Required** | `mydataset.zip`, `path/to/data/` |
| `--source` | Input source (predict only) | **Required** | `image.jpg` |
| `--im-size` | Input image size | 640 | Any positive integer |
| `--batch-size` | Batch size | 16 | Powers of 2 recommended |
| `--device` | Compute device | `cuda` | `cuda`, `cpu`, `mps` |
| `--workers` | Data loading workers | 4 | 0-16 recommended |
| `--output-dir` | Output directory | Auto-generated | Any valid path |

### Training-Specific Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--max-iters` | Training iterations | 3000 | Any positive integer |
| `--learning-rate` | Initial learning rate | 5e-4 | 1e-6 to 1e-1 |
| `--scheduler` | LR scheduler type | `MULTISTEP` | `MULTISTEP`, `COSINE`, `POLY`, `FIXED` |
| `--optimizer` | Optimizer type | `ADAMW` | `ADAMW`, `SGD`, `RMSPROP` |
| `--early-stop` | Enable early stopping | `true` | `true`, `false` |
| `--patience` | Early stopping patience | 10 | Any positive integer |
| `--ema-enabled` | Enable EMA | `false` | `true`, `false` |
| `--amp-enabled` | Enable mixed precision | `true` | `true`, `false` |

### Export Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--format` | Export format | `onnx` | `onnx`, `torchscript` |
| `--onnx-opset` | ONNX opset version | 17 | 11, 13, 16, 17 |
| `--overwrite` | Overwrite existing files | `false` | `true`, `false` |

### Prediction Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--conf` | Confidence threshold | 0.5 | 0.0 to 1.0 |
| `--save` | Save annotated images | `true` | `true`, `false` |
| `--save-json` | Save JSON results | `true` | `true`, `false` |
| `--save-masks` | Save segmentation masks | `true` | `true`, `false` |
| `--runtime` | Inference runtime | Auto | `onnx_cuda32`, `torchscript_32`, `onnx_cpu` |

## 📊 Dataset Layout
| Format | Description | Usage |
|--------|-------------|-------|
| `roboflow_coco` | Roboflow COCO format (Detection, Instance Segmentation) | `--dataset-layout roboflow_coco` |
| `roboflow_seg` | Roboflow segmentation format (Semantic Segmentation) | `--dataset-layout roboflow_seg` |
| `catalog` | Catalog format | `--dataset-layout catalog` |
| `cls_folder` | Classification folder format | `--dataset-layout cls_folder` |

## 🔧 Advanced Usage

### Environment Variables

Configure Focoos behavior through environment variables:

```bash
# Set default runtime type
export FOCOOS_RUNTIME_TYPE=onnx_cuda32

# Set logging level
export FOCOOS_LOG_LEVEL=INFO

# Set API key for Hub access
export FOCOOS_API_KEY=your_api_key
```


### Programmatic Usage

Use CLI commands programmatically in Python:

```python
from focoos.cli.commands import train_command, predict_command, export_command

# Train a model
train_command(
    model_name="fai-detr-m-coco",
    dataset_name="mydataset.zip",
    dataset_layout="roboflow_coco",
    im_size=640,
    run_name="my_training",
    max_iters=5000,
    batch_size=16
)

# Run inference
results = predict_command(
    model_name="fai-detr-m-coco",
    source="image.jpg",
    conf=0.5,
    save=True
)

# Export model
export_path = export_command(
    model_name="fai-detr-m-coco",
    format="onnx",
    device="cuda"
)
```

## 🐛 Troubleshooting

### Common Issues

**Command not found:**
```bash
# Ensure Focoos is installed
pip install focoos

# Check installation
focoos version
```

**CUDA out of memory:**
```bash
# Reduce batch size
focoos train --model fai-detr-m-coco --dataset data.zip --batch-size 8

# Use smaller image size
focoos train --model fai-detr-m-coco --dataset data.zip --im-size 480
```

**Dataset not found:**
```bash
# Check dataset path
ls -la mydataset.zip

# Use full path
focoos train --model fai-detr-m-coco --dataset /full/path/to/mydataset.zip
```

**Model loading errors:**
```bash
# Check available models
focoos hub models

# Verify model name
focoos train --model fai-detr-m-coco --dataset data.zip
```

### Getting Help

```bash
# Get general help
focoos --help

# Get command-specific help
focoos train --help
focoos predict --help
focoos export --help

# Check system compatibility
focoos checks

# View current settings
focoos settings
```

## 🚀 Performance Tips

### Training Optimization
- Use `--amp-enabled` for mixed precision training
- Increase `--workers` for faster data loading
- Use `--ema-enabled` for better model stability
- Enable `--early-stop` to prevent overfitting

### Inference Optimization
- Use ONNX runtime for faster inference: `--runtime onnx_cuda32`
- Choose appropriate `--im-size` for speed vs accuracy trade-off
- Use `--conf` threshold to filter low-confidence detections

### Memory Management
- Reduce `--batch-size` if experiencing OOM errors
- Use `--device cpu` for CPU-only processing
- Consider smaller `--im-size` for memory-constrained environments
