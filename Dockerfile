FROM python:3.12-slim-bullseye AS focoos-cpu
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
LABEL authors="focoos.ai"
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 gcc libmagic1 wget make cmake
WORKDIR /app
COPY focoos ./focoos
COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -e .[onnx-cpu]


FROM ghcr.io/focoosai/deeplearning:base-cu12-cudnn9-py312-uv AS focoos-gpu
LABEL authors="focoos.ai"

WORKDIR /app

COPY focoos ./focoos
COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -e .[onnx]


FROM ghcr.io/focoosai/deeplearning:cu12-cudnn9-py312-uv-tensorrt AS focoos-tensorrt
LABEL authors="focoos.ai"

WORKDIR /app

COPY focoos ./focoos
COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system -e .[onnx,tensorrt]
