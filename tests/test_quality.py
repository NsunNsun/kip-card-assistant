"""Проверки качества текста + регрессия на настоящих ответах модели.

tests/fixtures/first_run/*.json  — ответы openai/gpt-4o-mini с промптом версии 1;
tests/fixtures/second_run/*.json — с промптом версии 2.
Ошибки в них нашли при ручной вычитке; эти тесты гарантируют, что код ловит их сам.
"""

import json
import re
from pathlib import Path

import pytest

from kip_card.factcheck import FactChecker
from kip_card.generator import _normalize, _source_texts, generate_card, review
from kip_card.models import LLMCopy, ProductInput
from kip_card.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from kip_card.quality import evaluative_words, fix_title, generic_audience, irrelevant_queries
from kip_card.validation import parse_input

from .conftest import EXAMPLES, FakeLLM, make_copy

FIRST_RUN = Path(__file__).parent / "fixtures" / "first_run"
SECOND_RUN = Path(__file__).parent / "fixtures" / "second_run"


def product(name="ПД100-ДИ1,6-111-0,5", manufacturer="ОВЕН", product_type="датчик давления"):
    return ProductInput(name=name, manufacturer=manufacturer, product_type=product_type)


@pytest.mark.parametrize("title, expected", [
    ("датчик давления ПД100 ОВЕН", "Датчик давления ПД100 ОВЕН"),
    ("Пирометр testo 830-T1 Testo, -30 … +400 °C", "Пирометр testo 830-T1, -30 … +400 °C"),
    ("пирометр testo 830-T1; диапазон", "Пирометр testo 830-T1, диапазон"),
    ("testo 830-T1 пирометр", "testo 830-T1 пирометр"),  # написание производителя сохраняем
])
def test_fix_title(title, expected):
    p = product(name="testo 830-T1", manufacturer="Testo") if "testo" in title else product()
    assert fix_title(title, p) == expected


def test_irrelevant_queries():
    queries = ["датчик давления пд100-ди1,6-111-0,5", "датчик давления овен", "пирометр с лазерным целеуказателем",
               "датчики давлений купить", "купить недорого"]
    assert irrelevant_queries(queries, product()) == ["пирометр с лазерным целеуказателем", "купить недорого"]


def test_evaluative_words_allowed_only_if_in_source():
    source = "Обеспечивает надежное измерение давления"
    assert sorted(evaluative_words("Надёжный и идеальный прибор, самый точный", source)) == ["идеальный", "самый"]


def _replay(stem, run=FIRST_RUN):
    raw = (EXAMPLES / f"{stem}.json").read_text(encoding="utf-8")
    result = parse_input(raw)
    data = json.loads((run / f"{stem}.json").read_text(encoding="utf-8"))
    copy = _normalize(LLMCopy.model_validate(data), result.product)
    return copy, review(copy, FactChecker(_source_texts(result.product)), result.product, result.sparse)


def test_first_run_pd100_leaked_query_is_caught():
    copy, problems = _replay("owen-pd100-di1.6-111-0.5")
    assert copy.seo_title.startswith("Датчик давления")
    assert any("пирометр с лазерным целеуказателем" in p for p in problems)
    assert any("идеально" in p for p in problems)
    assert not any("надежн" in p.lower() for p in problems)  # «надежное» есть во входных данных


def test_first_run_testo_title_and_words():
    copy, problems = _replay("testo-830-t1")
    assert copy.seo_title == "Пирометр testo 830-T1, диапазон -30 … +400 °C"
    assert any("идеальным" in p for p in problems)


def test_first_run_manometer_words():
    _, problems = _replay("manometer-dm02-from-task")
    assert any("надежным" in p for p in problems)


def test_first_run_had_no_invented_values():
    for stem in ("testo-830-t1", "owen-pd100-di1.6-111-0.5", "manometer-dm02-from-task"):
        _, problems = _replay(stem)
        assert not any("значения, которых нет" in p for p in problems), stem


def test_leaked_query_triggers_repair(pd100_raw):
    bad = make_copy(title="Датчик давления ПД100-ДИ1,6-111-0,5 ОВЕН",
                    queries=["пд100-ди1,6-111-0,5", "датчик давления овен", "пирометр с лазерным целеуказателем"],
                    description=None)
    good = make_copy(title="Датчик давления ПД100-ДИ1,6-111-0,5 ОВЕН",
                     queries=["пд100-ди1,6-111-0,5", "датчик давления овен", "датчик давления 1,6 мпа"])
    # описание по умолчанию про пирометр — подменим на нейтральное для ПД100
    neutral = "Датчик давления ОВЕН ПД100-ДИ1,6-111-0,5 с верхним пределом 1,6 МПа.\n\n" + "Текст. " * 120
    bad["seo_description"] = good["seo_description"] = neutral[:950]
    llm = FakeLLM(bad, good)
    card = generate_card(pd100_raw, llm)
    assert card.meta.attempts == 2
    assert "пирометр" in llm.calls[1][-1]["content"]
    assert "пирометр с лазерным целеуказателем" not in card.search_queries


def test_system_prompt_has_no_concrete_products():
    # В версии 1 пример «пирометр с лазерным целеуказателем» попал в карточку датчика давления
    lowered = SYSTEM_PROMPT.lower()
    for word in ("testo", "пирометр", "пд100", "овен", "манометр", "830"):
        assert word not in lowered, word
    examples_with_digits = [m for m in re.findall(r"«[^»]*\d[^»]*»", SYSTEM_PROMPT) if m != "«№1»"]
    assert examples_with_digits == [], "в примерах промпта не должно быть конкретных значений"


def test_generic_audience():
    items = ["технические специалисты", "снабженцы", "инженеры КИПиА", "инженеры на производстве",
             "службы водоснабжения", "специалисты по энергоаудиту", "котельные"]
    assert generic_audience(items) == ["технические специалисты", "снабженцы", "инженеры КИПиА"]


def test_second_run_testo_is_fixed_and_roles_caught():
    copy, problems = _replay("testo-830-t1", SECOND_RUN)
    assert copy.seo_title == "Пирометр testo 830-T1, -30 … +400 °C"
    assert all(q == q.lower() for q in copy.search_queries)
    assert any("технические специалисты" in p and "снабженцы" in p for p in problems)


def test_second_run_pd100_is_clean():
    copy, problems = _replay("owen-pd100-di1.6-111-0.5", SECOND_RUN)
    assert problems == []
    assert "пд100-ди1,6-111-0,5 купить" in copy.search_queries


def test_sparse_input_does_not_demand_800_chars(manometer_raw):
    result = parse_input(manometer_raw)
    assert result.sparse
    short = make_copy(title="Манометр показывающий DM 02 Манотомь, 0–1,6 МПа",
                      description="Манометр показывающий DM 02 измеряет избыточное давление в диапазоне 0–1,6 МПа.",
                      queries=["манометр dm 02", "манометр dm 02 купить", "манометр манотомь"],
                      who_for=["системы, где контролируют избыточное давление"],
                      tasks=["измерение избыточного давления"])
    llm = FakeLLM(short)
    card = generate_card(result, llm)
    assert card.meta.attempts == 1  # короткий текст не отправлялся «добивать»
    assert "ДАННЫХ МАЛО" in llm.calls[0][1]["content"]
    assert any(i.field == "seo_description" and "мало сведений" in i.message for i in card.issues)
    assert card.status == "needs_review"
    assert card.meta.prompt_version == PROMPT_VERSION


def test_rich_input_still_requires_800_chars(testo_raw):
    llm = FakeLLM(make_copy(description="Коротко."), make_copy())
    card = generate_card(testo_raw, llm)
    assert card.meta.attempts == 2
    assert "ДАННЫХ МАЛО" not in llm.calls[0][1]["content"]
