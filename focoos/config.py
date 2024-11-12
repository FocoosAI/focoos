from typing import Optional

from pydantic_settings import BaseSettings

from focoos.ports import FocoosEnvHostUrl, RuntimeTypes


class FocoosConfig(BaseSettings):
    focoos_api_key: Optional[str] = None
    default_host_url: FocoosEnvHostUrl = FocoosEnvHostUrl.PROD
    runtime_type: RuntimeTypes = RuntimeTypes.ONNX_CUDA32
    warmup_iter: int = 2
