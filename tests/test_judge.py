"""Смысловая проверка (модель-судья). Модель и судья — заглушки, сеть не нужна.

Регрессия: tests/fixtures/third_run/*.json — ответы openai/gpt-4o-mini с промптом версии 3.
В тексте ПД100 «Корпус датчика выполнен из нержавеющей стали AISI 316L» — в данных из этой
стали сделана мембрана. Детерминированные проверки это пропускают; судья должен ловить.
"""

import json
from pathlib import Path

import pytest

from kip_card import cli
from kip_card.config import ConfigError, load_settings
from kip_card.factcheck import FactChecker
from kip_card.generator import _normalize, _source_texts, generate_card, review
from kip_card.judge import JUDGE_PROMPT, Judge, quote_found
from kip_card.llm import LLMError
from kip_card.models import LLMCopy
from kip_card.validation import parse_input

from .conftest import EXAMPLES, FakeLLM, make_copy

THIRD_RUN = Path(__file__).parent / "fixtures" / "third_run"
CORPUS = "Корпус датчика выполнен из нержавеющей стали AISI 316L"


def finding(quote, field="seo_description", reason="в данных этого нет", evidence="", verdict="not_in_data"):
    return {"field": field, "quote": quote, "reason": reason, "data_evidence": evidence, "verdict": verdict}


def report(*findings):
    return {"issues": list(findings)}


def third_run_copy(stem):
    raw = (EXAMPLES / f"{stem}.json").read_text(encoding="utf-8")
    data = json.loads((THIRD_RUN / f"{stem}.json").read_text(encoding="utf-8"))
    return raw, data


# ------------------------------------------------------------------ цитаты


@pytest.mark.parametrize("quote, text, expected", [
    ("корпус датчика выполнен", "Корпус  датчика\nвыполнен из стали", True),
    ("«в любых условиях».", "работает в любых условиях", True),
    ("от –40 до +100", "от -40 до +100 °С", True),
    ("Корпус из титана", "Корпус датчика выполнен из стали", False),
    ("  ", "любой текст", False),
])
def test_quote_found(quote, text, expected):
    assert quote_found(quote, text) is expected


# ------------------------------------------------------------------ сам судья


def test_judge_uses_own_schema_and_data_blocks(testo_raw):
    judge_llm = FakeLLM(report())
    copy = LLMCopy.model_validate(make_copy())
    assert Judge(judge_llm).review(parse_input(testo_raw).product, copy) == []
    assert judge_llm.schemas == ["judge_review"]
    user = judge_llm.calls[0][1]["content"]
    assert "<данные>" in user and "<тексты>" in user
    assert "12387" not in user  # идентификаторы судье тоже не передаются


def test_judge_drops_quotes_not_in_text(testo_raw):
    copy = LLMCopy.model_validate(make_copy())
    judge = Judge(FakeLLM(report(finding("этой фразы нет в тексте"), finding("лазерный целеуказатель упрощают наведение"))))
    found = judge.review(parse_input(testo_raw).product, copy)
    assert [f.quote for f in found] == ["лазерный целеуказатель упрощают наведение"]


def test_judge_drops_quotes_taken_from_input_data(testo_raw):
    """Фраза из данных не может быть выдумкой (ложное замечание из прогона gpt-4o-mini)."""
    copy = LLMCopy.model_validate(make_copy(title="Пирометр testo 830-T1 с лазерным целеуказателем"))
    judge = Judge(FakeLLM(report(finding("с лазерным целеуказателем", "seo_title", reason="усиление"))))
    assert judge.review(parse_input(testo_raw).product, copy) == []


def test_judge_bad_answer_is_llm_error(testo_raw):
    copy = LLMCopy.model_validate(make_copy())
    with pytest.raises(LLMError, match="схеме"):
        Judge(FakeLLM({"problems": []})).review(parse_input(testo_raw).product, copy)


def test_judge_prompt_has_no_concrete_products():
    lowered = JUDGE_PROMPT.lower()
    for word in ("testo", "пирометр", "пд100", "овен", "манометр", "aisi", "316", "изоляц", "двигател"):
        assert word not in lowered, word


# ------------------------------------------------------------------ в генераторе


def test_third_run_pd100_passes_deterministic_checks_but_not_judge(pd100_raw):
    """Ровно та ошибка, ради которой сделан судья: код её не видит."""
    raw, data = third_run_copy("owen-pd100-di1.6-111-0.5")
    result = parse_input(raw)
    copy = _normalize(LLMCopy.model_validate(data), result.product)
    assert review(copy, FactChecker(_source_texts(result.product)), result.product, result.sparse) == []
    assert CORPUS in copy.seo_description

    judge = Judge(FakeLLM(report(finding(CORPUS, reason="в данных из AISI 316L сделана мембрана"))))
    card = generate_card(result, FakeLLM(data), max_repair_attempts=0, judge=judge)
    assert card.status == "needs_review"
    assert card.meta.judge_status == "issues"
    assert any(i.message.startswith("Проверьте смысл: «Корпус датчика") for i in card.issues)


def test_judge_finding_triggers_repair(pd100_raw):
    _, bad = third_run_copy("owen-pd100-di1.6-111-0.5")
    good = dict(bad, seo_description=bad["seo_description"].replace(CORPUS, "Мембрана выполнена из нержавеющей стали AISI 316L"))
    llm = FakeLLM(bad, good)
    judge_llm = FakeLLM(report(finding(CORPUS, reason="в данных из AISI 316L сделана мембрана")), report())
    card = generate_card(pd100_raw, llm, judge=Judge(judge_llm))
    assert card.meta.attempts == 2
    assert card.status == "ok" and card.meta.judge_status == "passed"
    repair = llm.calls[1][-1]["content"]
    assert CORPUS in repair and "мембрана" in repair
    assert CORPUS not in card.seo_description
    assert len(judge_llm.calls) == 2  # исправленный текст тоже проверен


def test_judge_and_deterministic_problems_go_in_one_repair(testo_raw):
    bad = make_copy(description=make_copy()["seo_description"] + " Есть Bluetooth.")
    llm = FakeLLM(bad, make_copy())
    judge = Judge(FakeLLM(report(finding("лазерный целеуказатель упрощают наведение")), report()))
    card = generate_card(testo_raw, llm, judge=judge)
    repair = llm.calls[1][-1]["content"]
    assert "Bluetooth" in repair and "упрощают наведение" in repair
    assert card.status == "ok"


def test_judge_failure_does_not_break_card(testo_raw):
    judge = Judge(FakeLLM(LLMError("Не удалось подключиться к судье")))
    card = generate_card(testo_raw, FakeLLM(make_copy()), judge=judge)
    assert card.meta.judge_status == "error"
    assert card.status == "needs_review"
    assert any(i.field == "judge" and "не выполнена" in i.message for i in card.issues)


def test_judge_off(testo_raw):
    card = generate_card(testo_raw, FakeLLM(make_copy()))
    assert card.status == "ok"
    assert card.meta.judge_status == "off" and card.meta.judge_model is None


# ------------------------------------------------------------------ настройки и CLI


def test_judge_settings(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "gen-model")
    monkeypatch.setenv("JUDGE", "")
    settings = load_settings()
    assert settings.judge_enabled  # по умолчанию включён
    assert settings.for_judge().model == "gen-model" and settings.for_judge().temperature == 0.0
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    assert load_settings().for_judge().model == "judge-model"
    monkeypatch.setenv("JUDGE", "off")
    assert not load_settings().judge_enabled
    monkeypatch.setenv("JUDGE", "может быть")
    with pytest.raises(ConfigError, match="JUDGE"):
        load_settings()


def test_cli_generate_with_separate_judge_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("JUDGE", "on")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    gen, jud = FakeLLM(make_copy()), FakeLLM(report())
    jud.model_name = "judge-model"
    monkeypatch.setattr(cli, "OpenAICompatibleLLM", lambda s: jud if s.model == "judge-model" else gen)
    assert cli.main(["generate", str(EXAMPLES / "testo-830-t1.json"), "-o", str(tmp_path), "-f", "json"]) == cli.EXIT_OK
    meta = json.loads((tmp_path / "testo-830-t1.json").read_text(encoding="utf-8"))["meta"]
    assert meta["judge_model"] == "judge-model" and meta["judge_status"] == "passed"
    assert jud.schemas == ["judge_review"]


# ------------------------------------------------------------------ доказательства и вердикт (прогон версии 5)

V5_PD100 = Path(__file__).parent / "fixtures" / "v5_run" / "owen-pd100-di1.6-111-0.5.json"
SEWER = "Объекты ЖКХ, системы водоснабжения, водоподготовки и канализации"


def v5_pd100_copy():
    return LLMCopy.model_validate(json.loads(V5_PD100.read_text(encoding="utf-8")))


def test_confirmed_verdict_is_dropped(pd100_raw):
    """Ложное замечание из прогона: судья сам написал «Остальное подтверждается»."""
    judge = Judge(FakeLLM(report(finding("ПД100-ДИ1,6-111-0,5 ОВЕН, 1,6 МПа", "seo_title",
                                         reason="подтверждается", verdict="confirmed"))))
    assert judge.review(parse_input(pd100_raw).product, v5_pd100_copy()) == []


def test_real_evidence_is_shown_in_warning(pd100_raw):
    """Судья ошибся про канализацию — но приложенная цитата из данных сразу это показывает."""
    evidence = "мониторинг давления в сетях водоснабжения и канализации"
    judge = Judge(FakeLLM(report(finding(SEWER, "commercial", reason="канализация не названа", evidence=evidence))))
    card = generate_card(pd100_raw, FakeLLM(json.loads(V5_PD100.read_text(encoding="utf-8"))),
                         max_repair_attempts=0, judge=judge)
    warning = next(i.message for i in card.issues if i.message.startswith("Проверьте смысл"))
    assert f"(в данных: «{evidence}»)" in warning


def test_fake_evidence_is_not_shown(pd100_raw):
    judge = Judge(FakeLLM(report(finding(SEWER, "commercial", evidence="этой фразы нет в данных"))))
    [found] = judge.review(parse_input(pd100_raw).product, v5_pd100_copy())
    assert found.data_evidence == ""
    assert found.as_warning().endswith("(в данных об этом нет)")


def test_old_style_answer_without_new_fields_still_works(pd100_raw):
    judge = Judge(FakeLLM({"issues": [{"field": "commercial", "quote": SEWER, "reason": "нет в данных"}]}))
    [found] = judge.review(parse_input(pd100_raw).product, v5_pd100_copy())
    assert found.verdict == "not_in_data"


def test_judge_schema_requires_evidence_and_verdict():
    from kip_card.judge import JUDGE_SCHEMA
    item = JUDGE_SCHEMA.schema["properties"]["issues"]["items"]
    assert {"data_evidence", "verdict"} <= set(item["required"])
    assert "confirmed" in item["properties"]["verdict"]["enum"]
    assert "data_evidence" in JUDGE_PROMPT
