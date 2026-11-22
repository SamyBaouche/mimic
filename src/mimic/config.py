from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Empty = SQLite file in `home`. Or sqlite:///path/to/file.db, or postgresql://... (pgvector)
    database_url: str = ""
    mimic_home: Path = Path.home() / ".mimic"  # env: MIMIC_HOME
    ollama_host: str = "http://localhost:11434"
    # Local models. auto picks "standard" (7B) or "light" (3B) from this computer's memory;
    # VISION_MODEL / PLANNER_MODEL override one model. See mimic.hardware.
    mimic_profile: str = "auto"  # env: MIMIC_PROFILE = auto | standard | light
    vision_model: str = ""
    planner_model: str = ""
    embed_model: str = "nomic-embed-text"
    # Scans and photos have no text to check a reading against: read them twice, flag disagreement
    double_check_scans: bool = True  # env: DOUBLE_CHECK_SCANS
    profile_used: str = ""  # filled in below: the profile actually applied, and why
    profile_reason: str = ""
    sandbox_url: str = "http://localhost:8001"
    embedding_dim: int = 768
    # Published fine-tuned element matcher (downloaded once if you did not train your own).
    matcher_repo: str = "SamyBaouche/mimic-element-matcher"

    @model_validator(mode="after")
    def _models(self):
        from mimic.hardware import PROFILES, detect, recommend

        profile = self.mimic_profile.strip().lower()
        if profile not in PROFILES:
            if profile != "auto":
                raise ValueError(f"MIMIC_PROFILE must be auto, standard or light, not {profile!r}")
            if self.vision_model and self.planner_model:
                profile, reason = "custom", "VISION_MODEL and PLANNER_MODEL are set"
            else:
                profile, reason = recommend(detect())
        else:
            reason = "MIMIC_PROFILE"
        defaults = PROFILES.get(profile, PROFILES["standard"])
        self.vision_model = self.vision_model or defaults["vision_model"]
        self.planner_model = self.planner_model or defaults["planner_model"]
        self.profile_used, self.profile_reason = profile, reason
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
