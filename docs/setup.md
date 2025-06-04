# Python SDK Setup 🐍

## Install the Focoos SDK
The Focoos SDK can be installed with different package managers using python 3.10 and above.


We recommend using [UV](https://docs.astral.sh/uv/) (how to [install uv](https://docs.astral.sh/uv/getting-started/installation/)) as a package manager and environment manager for a streamlined dependency management experience.
however the installation process is the same if you use **pip** or **conda**.

You can easily create a new virtual environment with UV using the following command:
```bash linenums="0"
uv venv --python 3.12
source .venv/bin/activate
```

=== "Default"

    The default installation provides compatibility with both CPU and GPU environments, utilizing PyTorch as the default runtime.
    you can perfom training and inference with PyTorch

    ```bash linenums="0"
    uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos'
    ```

=== "ONNX Runtime (CUDA) "
    To perform inference using ONNX Runtime with GPU (CUDA) acceleration
    **Additional requirements:**
    Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
    To install cuDNN 9:

    ```bash linenums="0"
    apt-get -y install cudnn9-cuda-12
    ```

    ```bash linenums="0"
    uv pip install 'focoos[onnx] @ git+https://github.com/FocoosAI/focoos'
    ```

=== "ONNX Runtime (Tensorrt) "
    To perform inference using ONNX Runtime with GPU (Tensorrt) acceleration
    **Additional requirements:**
    Ensure that you have CUDA 12 and cuDNN 9 installed, as they are required for onnxruntime version 1.22.0.
    To install cuDNN 9:

    ```bash linenums="0"
    apt-get -y install cudnn9-cuda-12
    ```

    To perform inference using TensorRT, ensure you have TensorRT version 10.5 installed.

    ```bash linenums="0"
    uv pip install 'focoos[onnx,tensorrt] @ git+https://github.com/FocoosAI/focoos'
    ```

=== "CPU ONNX Runtime"

    ```bash linenums="0"
    uv pip install 'focoos[onnx-cpu] @ git+https://github.com/FocoosAI/focoos'
    ```

!!! note
    🤖 **Multiple Runtimes:** You can install multiple extras by running `pip install 'focoos[onnx,tensorrt] @ git+https://github.com/FocoosAI/focoos.git'`. Note that you can't use `onnx-cpu` and `onnx` or `tensorrt` at the same time.

!!! note
    🛠️ **Installation Tip:** If you want to install a specific version, for example `v0.14.1`, use:

    ```bash linenums="0"
    uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos@v0.14.1'
    ```

    📋 **Check Versions:** Visit [https://github.com/FocoosAI/focoos/tags](https://github.com/FocoosAI/focoos/tags) for available versions.


## Inference Runtime support
Focoos models support multiple inference runtimes. The library can be used without any extras for training and inference using the PyTorch runtime. Additional extras are only needed if you want to use ONNX or TensorRT runtimes for optimized inference.

| RuntimeType | Extra | Runtime | Compatible Devices | Available ExecutionProvider |
|------------|-------|---------|-------------------|---------------------------|
| TORCHSCRIPT_32 | - | torchscript | CPU, NVIDIA GPUs | - |
| ONNX_CUDA32 | `[onnx]` | onnxruntime GPU | NVIDIA GPUs | CUDAExecutionProvider |
| ONNX_TRT32 | `[tensorrt]` | onnxruntime TRT | NVIDIA GPUs (Optimized) | CUDAExecutionProvider, TensorrtExecutionProvider |
| ONNX_TRT16 | `[tensorrt]` | onnxruntime TRT | NVIDIA GPUs (Optimized) | CUDAExecutionProvider, TensorrtExecutionProvider |
| ONNX_CPU | `[onnx-cpu]` | onnxruntime CPU | CPU (x86, ARM), M1, M2, M3 (Apple Silicon) | CPUExecutionProvider, CoreMLExecutionProvider, AzureExecutionProvider |
| ONNX_COREML | `[onnx-cpu]` | onnxruntime CPU | M1, M2, M3 (Apple Silicon) | CoreMLExecutionProvider, CPUExecutionProvider |


## Docker and Devcontainers
For container support, Focoos offers different Docker images:

- `focoos-gpu`: Includes ONNX Runtime (CUDA) support
- `focoos-tensorrt`: Includes ONNX and TensorRT support
- `focoos-cpu`: only CPU

to use the docker images, you can run the following command:

```bash linenums="0"
docker build -t focoos-gpu . --target=focoos-gpu
docker run -it focoos-gpu
```

This repository also includes a devcontainer configuration for each of the above images. You can launch these devcontainers in Visual Studio Code for a seamless development experience.
