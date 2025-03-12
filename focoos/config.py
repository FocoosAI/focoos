"""
Configuration module for Focoos AI SDK.

This module defines the configuration settings for the Focoos AI SDK,
including API credentials, logging levels, default endpoints, and runtime preferences.

Classes:
    FocoosConfig: Pydantic settings class for Focoos SDK configuration.

Constants:
    LogLevel: Type definition for supported logging levels.
    FOCOOS_CONFIG: Global configuration instance.
"""

import typing
from typing import Optional

from pydantic_settings import BaseSettings

from focoos.ports import PROD_API_URL, RuntimeTypes

LogLevel = typing.Literal["DEBUG", "INFO", "WARNING", "ERROR", "FATAL", "CRITICAL"]


class FocoosConfig(BaseSettings):
    """
    Configuration settings for the Focoos AI SDK.

    This class uses pydantic_settings to manage configuration values,
    supporting environment variable overrides and validation. All attributes
    can be configured via environment variables with the same name.

    Attributes:
        focoos_api_key (Optional[str]): API key for authenticating with Focoos services.
            Defaults to None.
        focoos_log_level (LogLevel): Logging level for the SDK.
            Defaults to "DEBUG".
        default_host_url (str): Default API endpoint URL.
            Defaults to the production API URL.
        runtime_type (RuntimeTypes): Default runtime type for model inference.
            Defaults to ONNX_CUDA32 for NVIDIA GPU acceleration.
        warmup_iter (int): Number of warmup iterations for model initialization.
            Defaults to 2.

    Example:
        Setting configuration via environment variables in bash:
        ```bash
        # Set API key and change log level
        export FOCOOS_API_KEY="your-api-key-here"
        export FOCOOS_LOG_LEVEL="INFO"
        export DEFAULT_HOST_URL="https://custom-api-endpoint.com"
        export RUNTIME_TYPE="ONNX_CUDA32"
        export WARMUP_ITER="3"

        # Then run your Python application
        python your_app.py
        ```
    """

    focoos_api_key: Optional[str] = None
    focoos_log_level: LogLevel = "DEBUG"
    default_host_url: str = PROD_API_URL
    runtime_type: RuntimeTypes = RuntimeTypes.ONNX_CUDA32
    warmup_iter: int = 2


FOCOOS_CONFIG = FocoosConfig()
