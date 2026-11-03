import json

import pytest

from kip_card.validation import InputError, parse_input


def test_examples_detect_profiles(testo_raw, pd100_raw, manometer_raw):
    assert parse_input(testo_raw).profile == "pyrometer"
    assert parse_input(pd100_raw).profile == "pressure_sensor"
    assert parse_input(manometer_raw).profile == "manometer"


def test_complete_examples_have_no_missing_specs(testo_raw, pd100_raw):
    assert parse_input(testo_raw).missing_specs == []
    assert parse_input(pd100_raw).missing_specs == []


def test_manometer_from_task_is_flagged(manometer_raw):
    result = parse_input(manometer_raw)
    messages = " ".join(i.message for i in result.issues)
    assert result.missing_specs == ["Степень защиты"]
    assert "обрезанное" in messages
    assert "Короткое описание" in messages


@pytest.mark.parametrize("raw, expected", [
    ('{"name": "X",}', "Некорректный JSON"),
    ("", "Пустой ввод"),
    ("[1, 2]", "JSON-объект"),
    ('{"manufacturer": "ОВЕН"}', "Не указано название"),
    ('{"name": "   "}', "Не указано название"),
    ('{"name": "X", "specs": "диапазон 0-10"}', "specs должно быть объектом"),
    ('{"name": "X", "specs": [1, 2]}', "элементы списка"),
])
def test_invalid_input_raises(raw, expected):
    with pytest.raises(InputError) as err:
        parse_input(raw)
    assert expected in str(err.value)


def test_broken_json_reports_position():
    with pytest.raises(InputError) as err:
        parse_input('{\n  "name": "X"\n  "specs": {}\n}')
    assert "строка 3" in str(err.value)


def test_empty_values_become_missing():
    result = parse_input({"name": "ТРМ1", "specs": {"Напряжение питания": "", "Интерфейс": None, "Входы": "1"}})
    assert result.product.specs == {"Входы": "1"}
    assert result.empty_specs == ["Напряжение питания", "Интерфейс"]
    assert any("без значения" in i.message for i in result.issues)


def test_specs_as_list_of_pairs_and_types():
    result = parse_input({
        "name": "X",
        "specs": [{"name": "Класс точности", "value": 1.5}, {"name": "Есть дисплей", "value": True},
                  {"name": "Выходы", "value": ["4-20 мА", "RS-485"]}],
    })
    assert result.product.specs == {"Класс точности": "1.5", "Есть дисплей": "да", "Выходы": "4-20 мА; RS-485"}


def test_html_in_description_is_cleaned():
    result = parse_input({"name": "X", "description": "<p>Прибор&nbsp;для <b>измерения</b></p>"})
    assert result.product.description == "Прибор для измерения"


def test_aliases_and_unknown_fields():
    result = parse_input({"title": "X", "vendor": "ОВЕН", "price": 100})
    assert result.product.name == "X"
    assert result.product.manufacturer == "ОВЕН"
    assert any("price" in i.message and i.level == "info" for i in result.issues)


def test_identifiers_and_url_are_not_facts_for_llm(testo_raw):
    facts = parse_input(testo_raw).product.facts_for_llm()
    dumped = json.dumps(facts, ensure_ascii=False)
    assert "12387" not in dumped and "souz-pribor" not in dumped
    assert facts["название"] == "testo 830-T1"
