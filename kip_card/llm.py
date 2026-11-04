"""Клиент LLM через OpenAI-совместимый API.

Один и тот же код работает с OpenRouter, OpenAI, Ollama, vLLM и llama.cpp server:
меняется только LLM_BASE_URL в .env.

Структурированный ответ запрашивается по убыванию надёжности:
json_schema (строгая схема) → json_object (просто JSON) → prompt (JSON по инструкции).
Режим «auto» сам спускается по этой лестнице, если сервер или модель не поддерживает
более строгий вариант, и запоминает рабочий режим.

Рассуждающие модели (gpt-5.x и др.) часто не принимают temperature, а OpenRouter
с require_parameters из-за этого отклоняет и строгую схему. Поэтому перед спуском
со строгой схемы клиент один раз повторяет её без temperature.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import openai

from .config import Settings
from .prompts import JSON_INSTRUCTION

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Ошибка обращения к модели. Текст — для человека, на русском."""


class LLMResponseError(LLMError):
    """Модель ответила, но ответ не разобрать как JSON. Генератор попробует переспросить."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class JSONModel(Protocol):
    """Всё, что нужно генератору от модели. В тестах подменяется заглушкой."""

    model_name: str
    mode: str

    def generate(self, messages: list[dict], response_schema: "ResponseSchema | None" = None) -> tuple[dict, str]:
        """Возвращает (разобранный JSON, исходный текст ответа).

        response_schema — схема ответа; по умолчанию — карточка товара.
        """
        ...


# Схема ответа для режима json_schema (strict: все поля обязательны, лишних нет).
_STR_LIST = {"type": "array", "items": {"type": "string"}}
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["seo_title", "seo_description", "search_queries", "commercial", "data_gaps"],
    "properties": {
        "seo_title": {"type": "string", "description": "SEO-заголовок, до 70 символов"},
        "seo_description": {"type": "string", "description": "SEO-описание, 800–1200 символов"},
        "search_queries": {**_STR_LIST, "description": "3–5 поисковых запросов"},
        "commercial": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "who_for", "tasks"],
            "properties": {
                "summary": {"type": "string"},
                "who_for": _STR_LIST,
                "tasks": _STR_LIST,
            },
        },
        "data_gaps": _STR_LIST,
    },
}

MODES = ("json_schema", "json_object", "prompt")


@dataclass(frozen=True)
class ResponseSchema:
    """Схема ответа: строгая JSON-схема и текстовая подсказка для режимов без схемы."""

    name: str
    schema: dict
    json_hint: str


PRODUCT_CARD = ResponseSchema("product_card", RESPONSE_SCHEMA, JSON_INSTRUCTION)


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (в т.ч. обёрнутый в ```json ... ```)."""
    if not text or not text.strip():
        raise LLMResponseError("Модель вернула пустой ответ", text or "")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("В ответе модели нет JSON-объекта", text)
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Ответ модели — некорректный JSON: {exc.msg}", text) from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Ответ модели — JSON, но не объект", text)
    return data


def _human_error(exc: Exception, settings: Settings) -> LLMError:
    """Переводит исключения SDK в понятные сообщения."""
    where = settings.base_url
    if isinstance(exc, openai.AuthenticationError):
        return LLMError(f"Сервер {where} отклонил API-ключ (401). Проверьте LLM_API_KEY в .env")
    if isinstance(exc, openai.PermissionDeniedError):
        return LLMError(f"Нет доступа к модели {settings.model} (403): {exc.message}")
    if isinstance(exc, openai.NotFoundError):
        return LLMError(f"Модель {settings.model} не найдена на {where} (404). Проверьте LLM_MODEL")
    if isinstance(exc, openai.RateLimitError):
        return LLMError("Превышен лимит запросов (429). Подождите минуту и повторите")
    if isinstance(exc, openai.APITimeoutError):
        return LLMError(f"Модель не ответила за {settings.timeout:.0f} с. Увеличьте LLM_TIMEOUT или повторите")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(f"Не удалось подключиться к {where}. Проверьте адрес, сеть и что сервер модели запущен")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 402:
            return LLMError("Недостаточно средств на балансе провайдера (402). Пополните баланс OpenRouter")
        if exc.status_code >= 500:
            return LLMError(f"Сервер модели вернул ошибку {exc.status_code}. Повторите позже")
        return LLMError(f"Сервер модели вернул ошибку {exc.status_code}: {exc.message}")
    return LLMError(f"Ошибка обращения к модели: {exc}")


def _is_format_unsupported(exc: Exception) -> bool:
    """Похоже ли, что сервер/модель не поддерживает запрошенный response_format."""
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        return True
    if isinstance(exc, openai.NotFoundError):
        # OpenRouter: «No endpoints found that support the requested parameters».
        return "support" in str(exc).lower() or "parameter" in str(exc).lower()
    return False


def create_llm(settings: Settings | None = None) -> "OpenAICompatibleLLM":
    from .config import load_settings

    return OpenAICompatibleLLM(settings or load_settings())


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings, client: openai.OpenAI | None = None):
        self.settings = settings
        self.model_name = settings.model
        self.client = client or openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
            max_retries=2,  # SDK сам повторяет 429/5xx/обрывы с экспоненциальной паузой
            default_headers={"X-Title": "kip-card-assistant"} if settings.is_openrouter else None,
        )
        fmt = settings.response_format
        self._modes = list(MODES) if fmt == "auto" else [fmt]
        self.mode = self._modes[0]
        self._send_temperature = True
        self._tried_without_temperature = False

    def _request(self, messages: list[dict], mode: str, spec: ResponseSchema):
        kwargs: dict = {"model": self.settings.model, "messages": messages}
        if self._send_temperature:
            kwargs["temperature"] = self.settings.temperature
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": spec.name, "strict": True, "schema": spec.schema},
            }
            if self.settings.is_openrouter:
                # Маршрутизировать только к провайдерам, которые реально поддерживают схему.
                kwargs["extra_body"] = {"provider": {"require_parameters": True}}
        else:
            messages = [dict(m) for m in messages]
            messages[0]["content"] = messages[0]["content"] + spec.json_hint
            kwargs["messages"] = messages
            if mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def generate(self, messages: list[dict], response_schema: ResponseSchema | None = None) -> tuple[dict, str]:
        spec = response_schema or PRODUCT_CARD
        last_exc: Exception | None = None
        while self._modes:
            mode = self._modes[0]
            try:
                response = self._request(messages, mode, spec)
            except openai.APIError as exc:
                if _is_format_unsupported(exc) and mode == "json_schema" and not self._tried_without_temperature:
                    log.warning("Строгая схема отклонена (%s), повторяю без temperature", exc)
                    self._tried_without_temperature = True
                    self._send_temperature = False
                    last_exc = exc
                    continue
                if _is_format_unsupported(exc) and len(self._modes) > 1:
                    log.warning("Режим %s не поддерживается (%s), пробую следующий", mode, exc)
                    self._modes.pop(0)
                    self.mode = self._modes[0]
                    self._send_temperature = True  # схема не поддерживается сама по себе
                    last_exc = exc
                    continue
                raise _human_error(exc, self.settings) from exc
            except Exception as exc:  # неожиданное — тоже показываем по-человечески
                raise _human_error(exc, self.settings) from exc

            if not response.choices:
                raise LLMResponseError("Модель вернула ответ без вариантов (choices пуст)")
            choice = response.choices[0]
            text = choice.message.content or ""
            if choice.finish_reason == "length":
                raise LLMResponseError("Ответ модели обрезан по лимиту токенов", text)
            return extract_json(text), text

        raise _human_error(last_exc or RuntimeError("нет доступных режимов"), self.settings)
