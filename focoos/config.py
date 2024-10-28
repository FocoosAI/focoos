from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

from focoos.ports import FocoosEnvHostUrl


class FocoosCfg(BaseSettings):
    focoos_api_key: Optional[str] = None
    host_url: FocoosEnvHostUrl = FocoosEnvHostUrl.DEV
    deploy_max_wait: int = 10
    user_dir: Path = Path.home() / ".focoos"
