# Installation

The focoos SDK provides flexibility for installation based on the execution environment you plan to use. The package supports `CPU`, `NVIDIA GPU`, and `NVIDIA GPU with TensorRT` environments. Please note that only one execution environment should be selected during installation.

## Requirements

For **local inference**, ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.

To install cuDNN 9:

```bash linenums="0"
apt-get -y install cudnn9-cuda-12
```

To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.

## Installation Options

* CPU Environment

If you plan to run the SDK on a CPU-only environment:

```bash linenums="0"
pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU Environment

For execution using NVIDIA GPUs (with ONNX Runtime GPU support):

```bash linenums="0"
pip install 'focoos[gpu] @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU with TensorRT

For optimized execution using NVIDIA GPUs with TensorRT:

```bash linenums="0"
pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
```

!!! note
    🛠️ **Installation Tip:** If you want to install a specific version, for example `v0.1.3`, use:
    ```bash
    pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git@v0.1.3'
    ```
    📋 **Check Versions:** Visit [https://github.com/FocoosAI/focoos/tags](https://github.com/FocoosAI/focoos/tags) for available versions.
