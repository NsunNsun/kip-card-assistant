import json

import pytest

from kip_card.generator import TITLE_MAX, generate_card
from kip_card.llm import LLMError, LLMResponseError
from kip_card.models import NO_DATA
from kip_card.render import to_markdown
from kip_card.validation import parse_input

from .conftest import FakeLLM, make_copy, pad


def test_happy_path(testo_raw):
    llm = FakeLLM(make_copy())
    card = generate_card(testo_raw, llm)
    assert card.status == "ok"
    assert card.meta.attempts == 1
    assert 800 <= len(card.seo_description) <= 1200
    # Характеристики — дословно из входа, модель их не трогает
    source = json.loads(testo_raw)["specs"]
    assert {r.name: r.value for r in card.key_specs} == source
    assert all(r.source == "input" for r in card.key_specs)
    assert card.product["id:Код товара"] == "12387"


def test_prompt_contains_data_block_and_no_identifiers(testo_raw):
    llm = FakeLLM(make_copy())
    generate_card(testo_raw, llm)
    user_prompt = llm.calls[0][1]["content"]
    assert "<данные>" in user_prompt and "</данные>" in user_prompt
    assert "12387" not in user_prompt and "souz-pribor" not in user_prompt


def test_invented_value_triggers_repair(testo_raw):
    bad = make_copy(description=make_copy()["seo_description"] + " Есть интерфейс Bluetooth 5.0.")
    llm = FakeLLM(bad, make_copy())
    card = generate_card(testo_raw, llm)
    assert card.status == "ok"
    assert card.meta.attempts == 2
    repair_prompt = llm.calls[1][-1]["content"]
    assert "Bluetooth 5.0" in repair_prompt


def test_persistent_invention_becomes_warning(testo_raw):
    bad = make_copy(title="Пирометр testo 830-T1 до 1000 °C")
    card = generate_card(testo_raw, FakeLLM(bad), max_repair_attempts=1)
    assert card.status == "needs_review"
    assert any("1000" in i.message and i.level == "warning" for i in card.issues)


def test_invalid_json_then_valid(testo_raw):
    llm = FakeLLM(LLMResponseError("Ответ модели — некорректный JSON", "{oops"), make_copy())
    card = generate_card(testo_raw, llm)
    assert card.status == "ok" and card.meta.attempts == 2


def test_schema_mismatch_then_valid(testo_raw):
    broken = make_copy()
    del broken["commercial"]
    card = generate_card(testo_raw, FakeLLM(broken, make_copy()))
    assert card.meta.attempts == 2


def test_never_valid_raises(testo_raw):
    with pytest.raises(LLMError):
        generate_card(testo_raw, FakeLLM(LLMResponseError("пусто")), max_repair_attempts=1)


def test_long_title_is_truncated_with_warning(testo_raw):
    long_title = "Пирометр testo 830-T1 инфракрасный термометр с лазерным целеуказателем и подсветкой"
    assert len(long_title) > TITLE_MAX
    card = generate_card(testo_raw, FakeLLM(make_copy(title=long_title)), max_repair_attempts=0)
    assert len(card.seo_title) <= TITLE_MAX
    assert not card.seo_title.endswith(" ")
    assert any(i.field == "seo_title" for i in card.issues)


def test_short_description_warns(testo_raw):
    card = generate_card(testo_raw, FakeLLM(make_copy(description="Коротко.")), max_repair_attempts=0)
    assert card.status == "needs_review"
    assert any("seo_description" in i.field for i in card.issues)


def test_missing_specs_marked_no_data(manometer_raw):
    copy = make_copy(title="Манометр показывающий DM 02 Манотомь 0–1,6 МПа",
                     description=pad("Манометр показывающий DM 02 с диапазоном 0–1,6 МПа и классом точности 1,5."),
                     queries=["манометр dm 02", "манометр показывающий купить", "манометр 1,6 мпа"])
    llm = FakeLLM(copy)
    card = generate_card(manometer_raw, llm)
    rows = {r.name: r for r in card.key_specs}
    assert rows["Степень защиты"].value == NO_DATA and rows["Степень защиты"].source == "missing"
    assert "Степень защиты" in card.missing_fields
    assert card.status == "needs_review"  # описание обрезано и короткое
    assert "Степень защиты" in llm.calls[0][1]["content"]  # модели прямо сказали, чего нет


def test_empty_product_still_produces_card():
    raw = {"name": "ТРМ1-Щ1.У2.Р", "specs": {"Напряжение питания": ""}, "description": ""}
    copy = make_copy(title="Измеритель-регулятор ТРМ1-Щ1.У2.Р",
                     description=pad("Измеритель-регулятор ТРМ1-Щ1.У2.Р."),
                     queries=["трм1-щ1.у2.р", "измеритель-регулятор купить", "трм1 цена"])
    card = generate_card(raw, FakeLLM(copy))
    assert {"Производитель", "Описание", "Технические характеристики", "Напряжение питания"} <= set(card.missing_fields)
    names = [r.name for r in card.key_specs]
    assert names.count("Напряжение питания") == 1


def test_llm_error_propagates(testo_raw):
    with pytest.raises(LLMError, match="ключ"):
        generate_card(testo_raw, FakeLLM(LLMError("Сервер отклонил API-ключ")))


def test_data_gaps_become_info_notes(testo_raw):
    card = generate_card(testo_raw, FakeLLM(make_copy(data_gaps=["не указан класс защиты"])))
    assert card.status == "ok"
    assert any(i.level == "info" and "класс защиты" in i.message for i in card.issues)


def test_markdown_render(manometer_raw):
    copy = make_copy(title="Манометр показывающий DM 02",
                     description=pad("Манометр DM 02."),
                     queries=["манометр dm 02", "манометр купить", "манометр показывающий"])
    md = to_markdown(generate_card(parse_input(manometer_raw), FakeLLM(copy)))
    assert "## Требует внимания" in md
    assert "| Степень защиты | **данных нет** |" in md
    assert "## Поисковые запросы" in md
