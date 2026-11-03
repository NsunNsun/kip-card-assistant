from kip_card.factcheck import FactChecker
from kip_card.generator import _source_texts
from kip_card.validation import parse_input


def checker_for(raw):
    return FactChecker(_source_texts(parse_input(raw).product))


def test_values_from_input_pass_with_normalization(pd100_raw):
    fc = checker_for(pd100_raw)
    text = ("Верхний предел 1.6 МПа, класс точности 0,5 %, штуцер M20x1,5 (латиница), IP65, "
            "сигнал 4–20 мА, питание 12...36 В, наработка 500000 ч, мембрана AISI 316L, "
            "межповерочный интервал 2 года, ГОСТ Р 52931")
    assert fc.check(text) == []


def test_invented_values_are_caught(pd100_raw):
    fc = checker_for(pd100_raw)
    found = fc.check("Интерфейс RS-485 и HART, степень защиты IP67, давление до 16 бар, класс 0,25")
    assert found == ["RS-485", "IP67", "HART", "16", "0,25"]


def test_number_inside_code_not_reported_twice(testo_raw):
    fc = checker_for(testo_raw)
    assert fc.check("поддержка USB 2.0") == ["USB 2.0"]


def test_model_name_tokens_are_known(testo_raw):
    fc = checker_for(testo_raw)
    assert fc.check("пирометр testo 830-t1 купить, Testo 830 T1, оптика 10:1, вес 200 г") == []


def test_allowed_abbreviations(testo_raw):
    assert checker_for(testo_raw).check("SEO-описание для B2B") == []
