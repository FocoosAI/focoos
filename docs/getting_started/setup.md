# 🐍 Setup

Focoos models support multiple inference runtimes.
To keep the library lightweight and to allow users to use their environment, optional dependencies (e.g., torch, onnxruntime, tensorrt) are not installed by default.
Foocoos is shipped with the following extras dependencies:

- `[torch]`: torchscript CUDA
- `[cuda]`: onnxruntime CUDA
- `[tensorrt]`: onnxruntime TensorRT

!!! note
    🤖 **Multiple Runtimes:** You can install multiple extras by running `uv pip install .[torch,cuda,tensorrt]`.

## Requirements

For **local inference**, ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.

To install cuDNN 9:

```bash linenums="0"
apt-get -y install cudnn9-cuda-12
```

To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.

### UV

We recommend using [UV](https://docs.astral.sh/uv/) as a package manager and environment manager for a streamlined dependency management experience.
Here’s how to create a new virtual environment with UV:
```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
```

## Inference environments

* CPU Environment

If you plan to run the SDK on a CPU-only environment:

```bash linenums="0"
pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU Environment (torchscript)
```bash linenums="0"
pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
```

* NVIDIA GPU Environment (onnxruntime)

For execution using NVIDIA GPUs (with ONNX Runtime GPU support):

```bash linenums="0"
pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
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

## Docker and Devcontainers
For container support, Focoos offers four different Docker images:
- `focoos-cpu`: only CPU
- `focoos-cuda`: Includes ONNX (CUDA) support
- `focoos-torch`: Includes ONNX and Torchscript (CUDA) support
- `focoos-tensorrt`: Includes ONNX, Torchscript, and TensorRT  support

to use the docker images, you can run the following command:

```bash linenums="0"
docker run -it . --target=focoos-cpu
```

This repository also includes a devcontainer configuration for each of the above images. You can launch these devcontainers in Visual Studio Code for a seamless development experience.
