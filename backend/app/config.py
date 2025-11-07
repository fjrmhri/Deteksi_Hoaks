"""Configuration helpers for the FastAPI hoax detection service."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_name_or_path: str = Field(
        "model_terbaik",
        description=(
            "Path or Hugging Face identifier for the fine-tuned IndoBERT model. "
            "Override with the MODEL_NAME_OR_PATH environment variable when the "
            "model is stored elsewhere."
        ),
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        "auto",
        description="Device selection strategy. Use 'auto' to prefer CUDA when available.",
    )
    max_length: int = Field(
        256,
        ge=8,
        le=512,
        description="Maximum token length when encoding texts for IndoBERT.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""

    return Settings()
