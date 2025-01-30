FROM python:3.12-slim-bullseye AS focoos-cpu
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
LABEL authors="focoos.ai"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 gcc libmagic1 wget make cmake
WORKDIR /app
COPY focoos ./focoos
COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -e .


FROM ghcr.io/focoosai/deeplearning:base-cu12-cudnn9-py312-uv AS focoos-cuda
LABEL authors="focoos.ai"

WORKDIR /app

COPY focoos ./focoos
COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -e .[cuda]


FROM focoos-onnx AS focoos-torch
RUN uv pip install --system -e .[torch]

FROM focoos-torch AS focoos-tensorrt
RUN apt-get update && apt-get install -y \
    wget lsb-release && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb  && \
    apt-get update && apt-get install -y \
    tensorrt \
    python3-libnvinfer-dev \
    uff-converter-tf && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN uv pip install --system -e .[tensorrt]
