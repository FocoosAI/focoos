# Python SDK Setup 🐍

Focoos models support multiple inference runtimes. The library can be used without any extras for training and inference using the PyTorch runtime. Additional extras are only needed if you want to use ONNX or TensorRT runtimes for optimized inference.

| RuntimeType | Extra | Runtime | Compatible Devices | Available ExecutionProvider |
|------------|-------|---------|-------------------|---------------------------|
| TORCHSCRIPT_32 | - | torchscript | CPU, NVIDIA GPUs | - |
| ONNX_CUDA32 | `[onnx]` | onnxruntime GPU | NVIDIA GPUs | CUDAExecutionProvider |
| ONNX_TRT32 | `[tensorrt]` | onnxruntime TRT | NVIDIA GPUs (Optimized) | CUDAExecutionProvider, TensorrtExecutionProvider |
| ONNX_TRT16 | `[tensorrt]` | onnxruntime TRT | NVIDIA GPUs (Optimized) | CUDAExecutionProvider, TensorrtExecutionProvider |
| ONNX_CPU | `[onnx-cpu]` | onnxruntime CPU | CPU (x86, ARM), M1, M2, M3 (Apple Silicon) | CPUExecutionProvider, CoreMLExecutionProvider, AzureExecutionProvider |
| ONNX_COREML | `[onnx-cpu]` | onnxruntime CPU | M1, M2, M3 (Apple Silicon) | CoreMLExecutionProvider, CPUExecutionProvider |


## Install the Focoos SDK
The Focoos SDK can be installed with different package managers using python 3.10 and above.

=== "uv"
    We recommend using [UV](https://docs.astral.sh/uv/) (how to [install uv](https://docs.astral.sh/uv/getting-started/installation/)) as a package manager and environment manager for a streamlined dependency management experience.

    You can easily create a new virtual environment with UV using the following command:
    ```bash linenums="0"
    uv venv --python 3.12
    source .venv/bin/activate
    ```

    === "Torch Runtime"
        ```bash linenums="0"
        uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "CPU ONNX Runtime"
        ```bash linenums="0"
        uv pip install 'focoos[onnx-cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```

        ```bash linenums="0"
        uv pip install 'focoos[onnx] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime with TensorRT"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.
        ```bash linenums="0"
        uv pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

=== "pip"
    Create and activate a new virtual environment using pip with the following commands:
    ```bash linenums="0"
    python -m venv .venv
    source .venv/bin/activate
    ```
    === "Cloud Runtime"
        ```bash linenums="0"
        pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "CPU ONNX Runtime"
        ```bash linenums="0"
        pip install 'focoos[onnx-cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        ```bash linenums="0"
        pip install 'focoos[onnx] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime with TensorRT"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.
        ```bash linenums="0"
        pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

=== "conda"
    Create and activate a new [conda](https://docs.conda.io/en/latest/) (how to [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) environment with Python 3.10 or higher:
    ```bash linenums="0"
    conda create -n focoos python=3.12
    conda activate focoos
    conda install pip
    ```

    === "Cloud Runtime"
        ```bash linenums="0"
        pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "CPU ONNX Runtime"
        ```bash linenums="0"
        pip install 'focoos[onnx-cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        ```bash linenums="0"
        pip install 'focoos[onnx] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime with TensorRT"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.
        ```bash linenums="0"
        pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

!!! note
    🤖 **Multiple Runtimes:** You can install multiple extras by running `pip install 'focoos[onnx,tensorrt] @ git+https://github.com/FocoosAI/focoos.git'`. Note that you can't use `onnx-cpu` and `onnx` or `tensorrt` at the same time.

!!! note
    🛠️ **Installation Tip:** If you want to install a specific version, for example `v0.1.3`, use:
    ```bash linenums="0"
    pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git@v0.1.3'
    ```
    📋 **Check Versions:** Visit [https://github.com/FocoosAI/focoos/tags](https://github.com/FocoosAI/focoos/tags) for available versions.

## Docker and Devcontainers
For container support, Focoos offers different Docker images:

- `focoos-cpu`: only CPU
- `focoos-onnx`: Includes ONNX (CUDA) support
- `focoos-tensorrt`: Includes ONNX and TensorRT support

to use the docker images, you can run the following command:

```bash linenums="0"
docker build -t focoos-gpu . --target=focoos-gpu
docker run -it focoos-gpu
```

This repository also includes a devcontainer configuration for each of the above images. You can launch these devcontainers in Visual Studio Code for a seamless development experience.
