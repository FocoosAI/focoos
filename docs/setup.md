# Python SDK Setup 🐍

Focoos models support multiple inference runtimes.
To keep the library lightweight and to allow users to use their environment, optional dependencies (e.g., torch, onnxruntime, tensorrt) are not installed by default.

Focoos is shipped with the following local inference runtimes that requires to install additional dependencies. If you intend to use only Focoos AI servers for inference, you don't need to install any of the following dependencies.

| RuntimeType | Extra | Runtime | Compatible Devices | Available ExecutionProvider |
|------------|-------|---------|-------------------|---------------------------|
| ONNX_CUDA32 | `[cuda]` | onnxruntime CUDA | NVIDIA GPUs | CUDAExecutionProvider |
| ONNX_TRT32 | `[tensorrt]` | onnxruntime TRT | NVIDIA GPUs (Optimized) | CUDAExecutionProvider, TensorrtExecutionProvider |
| ONNX_TRT16 | `[tensorrt]` | onnxruntime TRT | NVIDIA GPUs (Optimized) | CUDAExecutionProvider, TensorrtExecutionProvider |
| ONNX_CPU | `[cpu]` | onnxruntime CPU | CPU (x86, ARM), M1, M2, M3 (Apple Silicon) | CPUExecutionProvider, CoreMLExecutionProvider, AzureExecutionProvider |
| ONNX_COREML | `[cpu]` | onnxruntime CPU | M1, M2, M3 (Apple Silicon) | CoreMLExecutionProvider, CPUExecutionProvider |
| TORCHSCRIPT_32 | `[torch]` | torchscript | CPU, NVIDIA GPUs | - |


## Install the Focoos SDK
The Focoos SDK can be installed with different package managers using python 3.10 and above.

=== "uv"
    We recommend using [UV](https://docs.astral.sh/uv/) as a package manager and environment manager for a streamlined dependency management experience.
    Here’s how to create a new virtual environment with UV:
    ```bash
    pip install uv
    uv venv --python 3.12
    source .venv/bin/activate
    ```

    === "Cloud Runtime"
        ```bash linenums="0"
        uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "CPU ONNX Runtime"
        ```bash linenums="0"
        uv pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "Torchscript Runtime"
        To run the models using the torchscript runtime, you need to install the torch package.
        ```bash linenums="0"
        uv pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```

        ```bash linenums="0"
        uv pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime with TensorRT"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
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
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
    === "Cloud Runtime"
        ```bash linenums="0"
        pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "CPU ONNX Runtime"
        ```bash linenums="0"
        pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "Torchscript Runtime"
        ```bash linenums="0"
        pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        ```bash linenums="0"
        pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime with TensorRT"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.
        ```bash linenums="0"
        pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

=== "conda"
    Create and activate a new [conda](https://docs.conda.io/en/latest/) environment with Python 3.10 or higher:
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
        pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "Torchscript Runtime"
        ```bash linenums="0"
        pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        ```bash linenums="0"
        pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "NVIDIA GPU ONNX Runtime with TensorRT"
        **Additional requirements:**
        Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
        To install cuDNN 9:
        ```bash linenums="0"
        apt-get -y install cudnn9-cuda-12
        ```
        To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.
        ```bash linenums="0"
        pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

!!! note
    🤖 **Multiple Runtimes:** You can install multiple extras by running `pip install .[torch,cuda,tensorrt]`. Anyway you can't use `cpu` and `cuda` or `tensorrt` at the same time.

!!! note
    🛠️ **Installation Tip:** If you want to install a specific version, for example `v0.1.3`, use:
    ```bash
    pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git@v0.1.3'
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
