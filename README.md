![Tests](https://github.com/FocoosAI/focoos/actions/workflows/test.yml/badge.svg??event=push&branch=main)

# Focoos pre-trained models

| Model Name          | Task                  | Metrics | Domain                          |
| ------------------- | --------------------- | ------- | ------------------------------- |
| focoos_object365    | Detection             | -       | Common Objects, 365 classes     |
| focoos_rtdetr       | Detection             | -       | Common Objects, 80 classes      |
| focoos_cts_medium   | Semantic Segmentation | -       | Autonomous driving, 30 classes  |
| focoos_cts_large    | Semantic Segmentation | -       | Autonomous driving, 30 classes  |
| focoos_ade_nano     | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_ade_small    | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_ade_medium   | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_ade_large    | Semantic Segmentation | -       | Common Scenes, 150 classes      |
| focoos_aeroscapes   | Semantic Segmentation | -       | Drone Aerial Scenes, 11 classes |
| focoos_isaid_nano   | Semantic Segmentation | -       | Satellite Imagery, 15 classes   |
| focoos_isaid_medium | Semantic Segmentation | -       | Satellite Imagery, 15 classes   |

# Focoos
Focoos is a comprehensive SDK designed for computer vision tasks such as object detection, semantic segmentation, instance segmentation, and more. It provides pre-trained models that can be easily integrated and customized by users for various applications.
Focoos supports both cloud and local inference, and enables training on the cloud, making it a versatile tool for developers working in different domains, including autonomous driving, common scenes, drone aerial scenes, and satellite imagery.

### Key Features

- **Pre-trained Models**: A wide range of pre-trained models for different tasks and domains.
- **Cloud Inference**: API to Focoos cloud inference.
- **Cloud Training**: Train custom models with the Focoos cloud.
- **Multiple Local Inference Runtimes**: Support for various inference runtimes including CPU, GPU, Torchscript CUDA, OnnxRuntime CUDA, and OnnxRuntime TensorRT.
- **Model Monitoring**: Monitor model performance and metrics.



# 🐍 Setup
We recommend using [UV](https://docs.astral.sh/uv/) as a package manager and environment manager for a streamlined dependency management experience.
Here’s how to create a new virtual environment with UV:
```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
```

Focoos models support multiple inference runtimes.
To keep the library lightweight, optional dependencies (e.g., torch, onnxruntime, tensorrt) are not installed by default.
You can install the required optional dependencies using the following syntax:

## CPU only or Remote Usage

```bash
uv pip install focoos[cpu] git+https://github.com/FocoosAI/focoos.git
```

## GPU Runtimes
### Torchscript CUDA
```bash
uv pip install focoos[torch] git+https://github.com/FocoosAI/focoos.git
```

### OnnxRuntime CUDA
ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.

```bash
apt-get -y install cudnn9-cuda-12
```

```bash
uv pip install focoos[onnx] git+https://github.com/FocoosAI/focoos.gi
```

### OnnxRuntime TensorRT

To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.
```bash
sudo apt-get install tensorrt
```

```bash
uv pip install focoos[tensorrt] git+https://github.com/FocoosAI/focoos.git
```


## 🤖 Cloud Inference

```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_remote_model("focoos_object365")
detections = model.infer("./image.jpg", threshold=0.4)
```

## 🤖 Cloud Inference with Gradio

setup FOCOOS_API_KEY_GRADIO environment variable with your Focoos API key

```bash
uv pip install focoos[gradio] git+https://github.com/FocoosAI/focoos.git
```

```bash
python gradio/app.py
```

## Local Inference

```python
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_local_model("focoos_object365")

detections = model.infer("./image.jpg", threshold=0.4)
```


## Docker and devcontainers
For container support, Focoos offers four different Docker images:
- `focoos-cpu`: only CPU
- `focoos-onnx`: Includes ONNX support
- `focoos-torch`: Includes ONNX and Torchscript support
- `focoos-tensorrt`: Includes ONNX, Torchscript, and TensorRT support

This repository also includes a devcontainer configuration for each of the above images. You can launch these devcontainers in Visual Studio Code for a seamless development experience.
