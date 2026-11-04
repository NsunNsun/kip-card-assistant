"""Регрессия на ответах реальных моделей из сравнения (промпт версии 3).

tests/fixtures/gpt54mini_run/*.json — openai/gpt-5.4-mini;
tests/fixtures/qwen_run/*.json      — qwen/qwen3.8-27b.
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
from kip_card.quality import restore_case, service_phrases, specs_in_audience, unnatural_queries
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
    for run in ("gpt54mini_run", "qwen_run"):
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
