"""
app/config.py — Centralised configuration loaded from .env

All other modules import settings from here.
Nothing is hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, read from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # VLM backend
    # ------------------------------------------------------------------ #
    vlm_backend: str = "ollama"          # "ollama" | "huggingface" | "remote"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava:7b"

    # ------------------------------------------------------------------ #
    # Data paths (relative to the backend/ working directory)
    # ------------------------------------------------------------------ #
    bigearthnet_data_dir: str = "data/bigearthnet"
    bigearthnet_annotations_file: str = ""
    output_dir: str = "outputs"

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Derived helpers (not env-vars)
    # ------------------------------------------------------------------ #
    @property
    def bigearthnet_path(self) -> Path:
        return Path(self.bigearthnet_data_dir)

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("log_level")
    @classmethod
    def _upper_log(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
