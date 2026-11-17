from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://mimic:mimic@localhost:5432/mimic"
    ollama_host: str = "http://localhost:11434"
    vision_model: str = "qwen2.5vl:7b"
    planner_model: str = "qwen2.5:7b"
    sandbox_url: str = "http://localhost:8001"
    embedding_dim: int = 768


@lru_cache
def get_settings() -> Settings:
    return Settings()
