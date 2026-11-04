"""Настройки сервиса из переменных окружения (и файла .env, если он есть).

Ключ API никогда не хранится в коде: он читается из LLM_API_KEY
(или OPENROUTER_API_KEY / OPENAI_API_KEY для удобства).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from urllib.parse import urlparse

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
RESPONSE_FORMATS = ("auto", "json_schema", "json_object", "prompt")
_ON = {"on", "1", "true", "yes", "да"}
_OFF = {"off", "0", "false", "no", "нет"}


class ConfigError(Exception):
    """Ошибка конфигурации: сервис нельзя запустить с такими настройками."""


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout: float = 90.0
    response_format: str = "auto"
    max_repair_attempts: int = 1
    judge_enabled: bool = True
    judge_model: str = ""  # пусто — та же модель, что и для генерации

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or ""

    @property
    def is_local(self) -> bool:
        """Локальный сервер (Ollama / vLLM / llama.cpp) — ключ не обязателен."""
        return self.host in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "ollama", "vllm", "llamacpp"}

    @property
    def is_openrouter(self) -> bool:
        return self.host.endswith("openrouter.ai")

    def for_judge(self) -> "Settings":
        """Настройки для смысловой проверки: своя модель (если задана) и температура 0."""
        return replace(self, model=self.judge_model or self.model, temperature=0.0)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть числом, получено: {raw!r}") from exc


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должно быть целым числом, получено: {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    raise ConfigError(f"{name} должно быть on или off, получено: {raw!r}")


def load_settings() -> Settings:
    """Читает настройки и проверяет их. Бросает ConfigError с понятным текстом."""
    load_dotenv(override=False)

    base_url = os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL
    api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    response_format = (os.getenv("LLM_RESPONSE_FORMAT", "").strip() or "auto").lower()
    if response_format not in RESPONSE_FORMATS:
        raise ConfigError(
            f"LLM_RESPONSE_FORMAT={response_format!r}; допустимо: {', '.join(RESPONSE_FORMATS)}"
        )

    settings = Settings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=_float("LLM_TEMPERATURE", 0.2),
        timeout=_float("LLM_TIMEOUT", 90.0),
        response_format=response_format,
        max_repair_attempts=max(0, _int("MAX_REPAIR_ATTEMPTS", 1)),
        judge_enabled=_bool("JUDGE", True),
        judge_model=os.getenv("JUDGE_MODEL", "").strip(),
    )

    if not settings.api_key or settings.api_key.endswith("..."):
        if settings.is_local:
            # Локальным серверам ключ не нужен, но SDK требует непустую строку.
            settings = Settings(**{**settings.__dict__, "api_key": "local"})
        else:
            raise ConfigError(
                "Не задан API-ключ. Скопируйте .env.example в .env и укажите LLM_API_KEY "
                f"(сервер: {settings.base_url})."
            )
    return settings
