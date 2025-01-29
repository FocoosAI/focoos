# Setup

The focoos SDK provides flexibility for installation based on the execution environment you plan to use. The package supports `CPU`, `NVIDIA GPU`, and `NVIDIA GPU with TensorRT` environments. Please note that only one execution environment should be selected during installation.

## Requirements

For **local inference**, ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.

To install cuDNN 9:

```bash linenums="0"
apt-get -y install cudnn9-cuda-12
```

To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.

## Inference environments

* CPU Environment

If you plan to run the SDK on a CPU-only environment:

```bash linenums="0"
pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU Environment (torchscript)
```bash linenums="0"
pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU Environment (onnxruntime)

For execution using NVIDIA GPUs (with ONNX Runtime GPU support):

```bash linenums="0"
pip install 'focoos[onnx] @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU with TensorRT (onnxruntime)

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

## Docker and devcontainers
For container support, Focoos offers four different Docker images:
- `focoos-cpu`: only CPU
- `focoos-onnx`: Includes ONNX support
- `focoos-torch`: Includes ONNX and Torchscript support
- `focoos-tensorrt`: Includes ONNX, Torchscript, and TensorRT support

to use the docker images, you can run the following command:

```bash linenums="0"
docker run -it . --target=focoos-cpu
```

This repository also includes a devcontainer configuration for each of the above images. You can launch these devcontainers in Visual Studio Code for a seamless development experience.
