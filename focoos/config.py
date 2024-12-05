import typing
from typing import Optional

from pydantic_settings import BaseSettings

from focoos.ports import PROD_API_URL, RuntimeTypes

LogLevel = typing.Literal["DEBUG", "INFO", "WARNING", "ERROR", "FATAL", "CRITICAL"]


class FocoosConfig(BaseSettings):
    focoos_api_key: Optional[str] = None
    focoos_log_level: LogLevel = "INFO"
    default_host_url: str = PROD_API_URL
    runtime_type: RuntimeTypes = RuntimeTypes.ONNX_CUDA32
    warmup_iter: int = 2


FOCOOS_CONFIG = FocoosConfig()
