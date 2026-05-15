"""
config.py  —  Application settings loaded from .env
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Minecraft
    mc_host: str    = "localhost"
    mc_port: int    = 25565
    mc_username: str = "MineAgent"
    mc_version: str = "1.20.4"

    # Backend server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Ollama (Phase 2 — not used in Day 1)
    ollama_url: str   = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct"


settings = Settings()
