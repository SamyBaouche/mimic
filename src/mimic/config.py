from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Empty = SQLite file in `home`. Or sqlite:///path/to/file.db, or postgresql://... (pgvector)
    database_url: str = ""
    mimic_home: Path = Path.home() / ".mimic"  # env: MIMIC_HOME
    ollama_host: str = "http://localhost:11434"
    vision_model: str = "qwen2.5vl:7b"
    planner_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    sandbox_url: str = "http://localhost:8001"
    embedding_dim: int = 768
    # Published fine-tuned element matcher (downloaded once if you did not train your own).
    matcher_repo: str = "SamyBaouche/mimic-element-matcher"


@lru_cache
def get_settings() -> Settings:
    return Settings()
