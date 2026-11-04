"""Общие фикстуры. Модель в тестах — заглушка: тесты не требуют ключа и сети."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "input"

# Разные нейтральные предложения без цифр, латиницы и оценок — ими добиваем длину описания.
# Предложения не повторяются: повтор смысла код считает проблемой.
FILLER_SENTENCES = [
    "Прибор подходит для оперативного контроля на объектах и в лабораториях.",
    "Работа с ним не требует специальной подготовки персонала.",
    "Корпус удобно держать одной рукой во время обхода оборудования.",
    "Показания выводятся на дисплей сразу после наведения на поверхность.",
    "Перед выездом достаточно проверить заряд элемента питания.",
    "Сервисные службы берут его на плановые осмотры тепловых сетей.",
    "Энергетики используют такие устройства при обследовании щитов.",
    "Результат удобно сравнивать с предыдущими замерами в журнале.",
    "Комплект поставки помещается в небольшой кейс или сумку.",
    "Устройство хранится в сухом месте вдали от источников влаги.",
    "Наладчики применяют его при пуске новых систем вентиляции.",
    "Замер выполняется без контакта с объектом и без остановки линии.",
]
FILLER = " ".join(FILLER_SENTENCES)


def pad(first: str, target: int = 900) -> str:
    """Описание нужной длины: первый абзац + разные нейтральные предложения без цифр."""
    text = first + "\n\n"
    for sentence in FILLER_SENTENCES:
        if len(text) + len(sentence) + 1 > target:
            break
        text += sentence + " "
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
        self.schemas: list[str] = []

    def generate(self, messages, response_schema=None):
        self.calls.append(messages)
        self.schemas.append(response_schema.name if response_schema else "product_card")
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
                 "LLM_RESPONSE_FORMAT", "LLM_TEMPERATURE", "LLM_TIMEOUT", "MAX_REPAIR_ATTEMPTS", "JUDGE_MODEL"):
        monkeypatch.setenv(name, "")
    # Судья по умолчанию включён; в тестах выключаем, где он не нужен, чтобы не было сети
    monkeypatch.setenv("JUDGE", "off")
