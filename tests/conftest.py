"""Общие фикстуры. Модель в тестах — заглушка: тесты не требуют ключа и сети."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "input"

# Абзац без цифр и латинских кодов — им добиваем длину описания до нужной.
FILLER = (
    "Прибор подходит для оперативного контроля на объектах и в лабораториях, "
    "прост в использовании и не требует специальной подготовки персонала. "
)


def pad(first: str, target: int = 900) -> str:
    """Описание нужной длины: первый абзац + нейтральный текст без цифр."""
    text = first + "\n\n"
    while len(text) + len(FILLER) <= target:
        text += FILLER
    return text.strip()


def make_copy(title="Пирометр testo 830-T1 с лазерным целеуказателем",
              description=None, queries=None, summary="Пирометр для бесконтактного контроля температуры.",
              who_for=None, tasks=None, data_gaps=None) -> dict:
    if description is None:
        description = pad(
            "Пирометр testo 830-T1 измеряет температуру поверхностей в диапазоне -30 … +400 °C. "
            "Оптика 10:1 и лазерный целеуказатель упрощают наведение."
        )
    return {
        "seo_title": title,
        "seo_description": description,
        "search_queries": queries or ["пирометр testo 830-t1", "инфракрасный термометр купить", "пирометр с лазерным целеуказателем"],
        "commercial": {
            "summary": summary,
            "who_for": who_for or ["энергетики", "сервисные инженеры"],
            "tasks": tasks or ["контроль нагрева двигателей", "обследование изоляции"],
        },
        "data_gaps": data_gaps or [],
    }


class FakeLLM:
    """Возвращает заранее заданные ответы по очереди и запоминает запросы."""

    model_name = "fake-model"
    mode = "json_schema"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append(messages)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item, json.dumps(item, ensure_ascii=False)


@pytest.fixture
def testo_raw() -> str:
    return (EXAMPLES / "testo-830-t1.json").read_text(encoding="utf-8")


@pytest.fixture
def pd100_raw() -> str:
    return (EXAMPLES / "owen-pd100-di1.6-111-0.5.json").read_text(encoding="utf-8")


@pytest.fixture
def manometer_raw() -> str:
    return (EXAMPLES / "manometer-dm02-from-task.json").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Тесты не должны зависеть от .env разработчика."""
    for name in ("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                 "LLM_RESPONSE_FORMAT", "LLM_TEMPERATURE", "LLM_TIMEOUT", "MAX_REPAIR_ATTEMPTS"):
        monkeypatch.setenv(name, "")
