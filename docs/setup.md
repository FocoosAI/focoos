# Python SDK Setup 🐍

Focoos models support multiple inference runtimes.
To keep the library lightweight and to allow users to use their environment, optional dependencies (e.g., torch, onnxruntime, tensorrt) are not installed by default.

Focoos is shipped with the following extras dependencies for local inference:

- `[cpu]`: onnxruntime CPU
- `[cuda]`: onnxruntime CUDA
- `[torch]`: torchscript
- `[tensorrt]`: onnxruntime TensorRT


## Install the Focoos SDK

We recommend using [UV](https://docs.astral.sh/uv/) as a package manager and environment manager for a streamlined dependency management experience.
Here’s how to create a new virtual environment with UV:
```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
```

=== "Remote Runtime"
    If you plan to run the SDK on a remote machine, you can install the SDK without any extras.

    === "uv"
        ```bash linenums="0"
        uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "pip"
        ```bash linenums="0"
        pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "conda"
        ```bash linenums="0"
        conda install pip
        pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
        ```

=== "CPU ONNX Runtime"
    If you plan to run the SDK on a CPU-only environment.

    === "uv"
        ```bash linenums="0"
        uv pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "pip"
        ```bash linenums="0"
        pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "conda"
        ```bash linenums="0"
        conda install pip
        pip install 'focoos[cpu] @ git+https://github.com/FocoosAI/focoos.git'
        ```

=== "Torchscript Runtime"
    To run the models using the torchscript runtime, you need to install the torch package.
    The torch package is not installed by default, as it is not required for the CPU ONNX Runtime.

    === "uv"
        ```bash linenums="0"
        uv pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "pip"
        ```bash linenums="0"
        pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "conda"
        ```bash linenums="0"
        conda install pip
        pip install 'focoos[torch] @ git+https://github.com/FocoosAI/focoos.git'
        ```

=== "NVIDIA GPU ONNX Runtime"
    For execution using NVIDIA GPUs (with ONNX Runtime GPU support).

    === "uv"
        ```bash linenums="0"
        uv pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "pip"
        ```bash linenums="0"
        pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "conda"
        ```bash linenums="0"
        conda install pip
        pip install 'focoos[cuda] @ git+https://github.com/FocoosAI/focoos.git'
        ```
    **Additional requirements:**
    Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
    To install cuDNN 9:

    ```bash linenums="0"
    apt-get -y install cudnn9-cuda-12
    ```

=== "NVIDIA GPU ONNX Runtime with TensorRT"
    For optimized execution using NVIDIA GPUs with TensorRT.

    === "uv"
        ```bash linenums="0"
        uv pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "pip"
        ```bash linenums="0"
        pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    === "conda"
        ```bash linenums="0"
        conda install pip
        pip install 'focoos[tensorrt] @ git+https://github.com/FocoosAI/focoos.git'
        ```

    **Additional requirements:**
    Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.20.1.
    To install cuDNN 9:

    ```bash linenums="0"
    apt-get -y install cudnn9-cuda-12
    ```

    To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.

!!! note
    🤖 **Multiple Runtimes:** You can install multiple extras by running `pip install .[torch,cuda,tensorrt]`.

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
