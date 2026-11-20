"""Settings loaded from config.yaml, overridable with CSVX_<KEY> environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SAMPLES_DIR = PROJECT_ROOT / "samples"


class Settings(BaseModel):
    model: str = "openai:gpt-4.1-mini"
    temperature: float | None = 0.1
    agent_max_requests: int = 12
    history_turns: int = 6
    table_rows_to_model: int = 40
    max_upload_mb: int = 200
    categorical_max_unique: int = 60
    na_values: list[str] = ["", "NA", "N/A", "n/a", "na", "#N/A", "#NA", "<NA>", "null", "NULL", "?", "-", "--", "nan", "NaN"]
    max_chart_points: int = 20000
    max_color_series: int = 8
    semantic_search: bool = False
    embedding_model: str = "text-embedding-3-small"


def _coerce(raw: str, current: object) -> object:
    """Turn an environment-variable string into the type of the current value."""
    if raw.lower() in {"null", "none"}:
        return None
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [part.strip() for part in raw.split(",")]
    return raw


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_PATH
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = Settings(**data).model_dump()
    for key, value in defaults.items():
        env = os.environ.get(f"CSVX_{key.upper()}")
        if env is not None:
            data[key] = _coerce(env, value if value is not None else 0.0)
    return Settings(**data)


def provider_of(model: str) -> str:
    """'openai:gpt-4.1-mini' -> 'openai'."""
    return model.split(":", 1)[0] if ":" in model else "openai"


API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def missing_api_key(model: str) -> str | None:
    """Name of the environment variable the model needs but that is unset, else None."""
    var = API_KEY_VARS.get(provider_of(model))
    if var and not os.environ.get(var):
        return var
    return None
