FROM ghcr.io/focoosai/deeplearning:base-cu12-pt22
COPY ./pyproject.toml /app/pyproject.toml
RUN pip install ray[serve]
WORKDIR /app
COPY ./focoos /app/focoos
COPY ./ray-serve.py /app/ray-serve.py
COPY ./ray-config.yaml /app/ray-config.yaml
RUN pip install ".[inference-gpu,dev]"
RUN pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ --force-reinstall
COPY ./apps /app/apps
