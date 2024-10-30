from pydantic import BaseModel

from backend.config.loading import config


class ApiConfig(BaseModel):
    host: str
    port: int
    reload: bool


API_CONFIG = ApiConfig(**config["api_config"])
