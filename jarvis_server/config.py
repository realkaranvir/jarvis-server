import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, PositiveFloat, model_validator


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class ServiceConfig(BaseModel):
    provider: Literal["openai-compatible"] = "openai-compatible"
    base_url: str
    model: str
    timeout_seconds: PositiveFloat = 120
    api_key: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


class STTConfig(ServiceConfig):
    language: str | None = None


class VADConfig(BaseModel):
    mode: int = 2
    frame_duration_ms: Literal[10, 20, 30] = 30
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 600
    min_speech_duration_ms: int = 300
    max_speech_duration_seconds: int = 60


class LLMConfig(BaseModel):
    default: str
    servers: dict[str, ServiceConfig]

    @model_validator(mode="after")
    def default_backend_exists(self) -> "LLMConfig":
        if self.default not in self.servers:
            raise ValueError(f"Default LLM backend '{self.default}' is not configured")
        return self


class AppConfig(BaseModel):
    llm: LLMConfig
    stt: STTConfig
    vad: VADConfig = VADConfig()


def get_config_path() -> Path:
    return Path(os.environ.get("JARVIS_CONFIG", DEFAULT_CONFIG_PATH))


def load_config(path: Path) -> AppConfig:
    return AppConfig.model_validate_json(path.read_text())
