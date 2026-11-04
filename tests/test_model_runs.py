"""Регрессия на ответах реальных моделей.

tests/fixtures/gpt54mini_run/*.json — openai/gpt-5.4-mini, промпт версии 3;
tests/fixtures/qwen_run/*.json      — qwen/qwen3.8-27b, промпт версии 3;
tests/fixtures/v4_run/*.json        — qwen/qwen3.8-27b + судья gpt-5.4-mini, промпт версии 4;
tests/fixtures/v5_run/*.json        — то же, промпт версии 5.
Каждая проблема, найденная при ручной вычитке, должна ловиться или исправляться кодом,
а хорошая карточка (ПД100 от Qwen) — проходить проверки без замечаний.
"""

import json
from pathlib import Path

import pytest

from kip_card import cli
from kip_card.factcheck import FactChecker
from kip_card.generator import _normalize, _source_texts, review
from kip_card.models import LLMCopy, ProductInput
from kip_card.generator import WHO_FOR_MISSING, generate_card
from kip_card.judge import JUDGE_PROMPT
from kip_card.quality import (abbreviations, irrelevant_queries, repeated_sentences, restore_case, service_phrases,
                              specs_in_audience, unnatural_queries)
from kip_card.render import to_markdown
from kip_card.validation import parse_input

from .conftest import EXAMPLES, FakeLLM, make_copy

FIXTURES = Path(__file__).parent / "fixtures"
PD100 = "owen-pd100-di1.6-111-0.5"
TESTO = "testo-830-t1"
MANOMETER = "manometer-dm02-from-task"


def replay(run, stem):
    result = parse_input((EXAMPLES / f"{stem}.json").read_text(encoding="utf-8"))
    data = json.loads((FIXTURES / run / f"{stem}.json").read_text(encoding="utf-8"))
    copy = _normalize(LLMCopy.model_validate(data), result.product)
    return copy, review(copy, FactChecker(_source_texts(result.product)), result.product, result.sparse)


def has(problems, *parts):
    return any(all(part in p for part in parts) for p in problems)


# ------------------------------------------------------------------ Qwen


def test_qwen_pd100_is_clean_and_brand_case_restored():
    copy, problems = replay("qwen_run", PD100)
    assert problems == []
    assert copy.seo_description.startswith("Датчик давления ОВЕН ПД100-ДИ1,6-111-0,5")
    assert "пд100-ди1,6-111-0,5 купить" in copy.search_queries  # запросы остаются строчными


def test_qwen_manometer_filler_and_specs_in_audience():
    _, problems = replay("qwen_run", MANOMETER)
    assert has(problems, "служебные обороты", "включают")
    assert has(problems, "who_for", "G1/2")


def test_qwen_testo_service_phrase():
    _, problems = replay("qwen_run", TESTO)
    assert has(problems, "служебные обороты", "По описанию")


# ------------------------------------------------------------------ gpt-5.4-mini


def test_gpt54_manometer():
    _, problems = replay("gpt54mini_run", MANOMETER)
    assert has(problems, "служебные обороты", "По данным")
    assert has(problems, "who_for", "0–1,6 МПа")


def test_gpt54_pd100():
    _, problems = replay("gpt54mini_run", PD100)
    assert has(problems, "служебные обороты", "указанным характеристикам")
    assert has(problems, "search_queries", "4...20")
    assert has(problems, "who_for", "4...20 мА")


def test_gpt54_testo():
    _, problems = replay("gpt54mini_run", TESTO)
    assert has(problems, "служебные обороты", "ищут как")
    assert has(problems, "search_queries", "+400 °c")


def test_no_run_has_invented_values():
    for run in ("gpt54mini_run", "qwen_run", "v4_run", "v5_run"):
        for stem in (PD100, TESTO, MANOMETER):
            _, problems = replay(run, stem)
            assert not has(problems, "значения, которых нет"), (run, stem)


# ------------------------------------------------------------------ функции по отдельности


@pytest.mark.parametrize("name, maker, text, expected", [
    ("ПД100-ДИ1,6-111-0,5", "ОВЕН", "датчик овен пд100-ди1,6-111-0,5.", "датчик ОВЕН ПД100-ДИ1,6-111-0,5."),
    ("testo 830-T1", "Testo", "пирометр testo 830-t1", "пирометр testo 830-T1"),  # testo строчными — как в данных
    ("Манометр показывающий DM 02", "Манотомь", "манометр dm 02 манотомь", "манометр DM 02 Манотомь"),
    ("ПД100-ДИ1,6-111-0,5", "ОВЕН", "Пд100 и пд100-ди1,6-111-0,5x", "Пд100 и пд100-ди1,6-111-0,5x"),
])
def test_restore_case(name, maker, text, expected):
    assert restore_case(text, ProductInput(name=name, manufacturer=maker)) == expected


def test_service_phrases_do_not_fire_on_normal_text():
    text = ("Датчик измеряет избыточное давление и применяется в котельной автоматике. "
            "Характеристики стабильны во времени.")
    assert service_phrases(text) == []


def test_unnatural_queries_and_specs_in_audience():
    assert unnatural_queries(["пирометр testo", "пирометр -30 … +400 °c", "датчик 4...20 ма"]) == [
        "пирометр -30 … +400 °c", "датчик 4...20 ма"]
    assert specs_in_audience(["котельная автоматика", "системы с входом 4...20 мА"]) == ["системы с входом 4...20 мА"]


# ------------------------------------------------------------------ CLI


def test_cli_prints_elapsed_time(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(cli, "OpenAICompatibleLLM", lambda settings: FakeLLM(make_copy()))
    cli.main(["generate", str(EXAMPLES / f"{TESTO}.json"), "-o", str(tmp_path)])
    assert " с · сохранено" in capsys.readouterr().err


def test_cli_ctrl_c(monkeypatch, capsys):
    def interrupted(argv):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_main", interrupted)
    assert cli.main(["check", "x"]) == 130
    assert "Ctrl+C" in capsys.readouterr().err


# ------------------------------------------------------------------ прогон версии 4 (Qwen + судья gpt-5.4-mini)


def test_v4_pd100_is_clean():
    _, problems = replay("v4_run", PD100)
    assert problems == []


def test_v4_manometer_repeat_caught_and_short_model_query_ok():
    copy, problems = replay("v4_run", MANOMETER)
    assert has(problems, "повтор уже сказанного", "Прибор относится к показывающим манометрам")
    assert not has(problems, "не про этот товар")  # «dm 02» — модель манометра (была ложная пометка)


def test_v4_testo_abbreviation_caught():
    _, problems = replay("v4_run", TESTO)
    assert has(problems, "нераскрытые сокращения", "изм. знач.")


@pytest.mark.parametrize("query, relevant", [
    ("dm 02", True), ("манометр dm 02", True), ("dm", False), ("купить недорого", False),
])
def test_short_model_queries(query, relevant):
    product = ProductInput(name="Манометр показывающий DM 02", manufacturer="Манотомь")
    assert (irrelevant_queries([query], product) == []) is relevant


def test_repeated_sentences_and_abbreviations_do_not_fire_on_good_texts():
    for run, stem in (("v4_run", PD100), ("qwen_run", PD100), ("third_run", PD100), ("v4_run", TESTO)):
        data = json.loads((FIXTURES / run / f"{stem}.json").read_text(encoding="utf-8"))
        assert repeated_sentences(data["seo_description"]) == [], (run, stem)
    assert abbreviations("Погрешность ±1,5 °C. Корпус 190 x 75 x 38 мм. Используется батарейка.") == []


# ------------------------------------------------------------------ «Кому подходит: данных нет»


def test_sparse_data_allows_empty_who_for_as_no_data(manometer_raw):
    copy = make_copy(title="Манометр показывающий DM 02 Манотомь",
                     description="Манометр показывающий DM 02 измеряет избыточное давление в диапазоне 0–1,6 МПа.",
                     queries=["манометр dm 02", "манометр dm 02 купить", "манометр манотомь"],
                     tasks=["измерение избыточного давления"])
    copy["commercial"]["who_for"] = []
    llm = FakeLLM(copy)
    card = generate_card(manometer_raw, llm)
    assert card.meta.attempts == 1
    assert WHO_FOR_MISSING in card.missing_fields
    assert "**Кому подходит:**\n- **данных нет**" in to_markdown(card)
    assert "пустой список" in llm.calls[0][1]["content"]  # модели сказано, что так можно


def test_rich_data_requires_who_for(testo_raw):
    empty = make_copy()
    empty["commercial"]["who_for"] = []
    card = generate_card(testo_raw, FakeLLM(empty, make_copy()))
    assert card.meta.attempts == 2
    assert WHO_FOR_MISSING not in card.missing_fields


def test_judge_prompt_title_and_synonym_rules():
    assert "в заголовке проверяй только факты" in JUDGE_PROMPT
    assert "К seo_title это не относится" in JUDGE_PROMPT
    assert "синонимичный пересказ" in JUDGE_PROMPT


# ------------------------------------------------------------------ прогон версии 5: все три — без замечаний кода


@pytest.mark.parametrize("stem", [PD100, TESTO, MANOMETER])
def test_v5_run_passes_deterministic_checks(stem):
    copy, problems = replay("v5_run", stem)
    assert problems == [], problems
    if stem == MANOMETER:
        assert copy.commercial.who_for == []  # честное «данных нет» вместо тавтологии
