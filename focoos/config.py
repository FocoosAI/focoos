from typing import Optional

from pydantic_settings import BaseSettings

from focoos.ports import PROD_API_URL, RuntimeTypes


class FocoosConfig(BaseSettings):
    focoos_api_key: Optional[str] = None
    default_host_url: str = PROD_API_URL
    runtime_type: RuntimeTypes = RuntimeTypes.ONNX_CUDA32
    warmup_iter: int = 2
