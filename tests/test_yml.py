from pathlib import Path

import pytest

from kip_card.validation import parse_input
from kip_card.yml_import import FeedError, audit, find_offers, load_feed

FIXTURE = Path(__file__).parent / "fixtures" / "feed_excerpt.xml"

FEED_WITH_PARAMS = """<?xml version="1.0" encoding="UTF-8"?>
<yml_catalog date="2026-11-03"><shop><name>Test</name>
<categories><category id="1">КИП</category><category id="2" parentId="1">Манометры</category></categories>
<offers><offer id="7" available="true">
  <name>ДМ 02</name><typePrefix>манометр</typePrefix><vendor>Манотомь</vendor><categoryId>2</categoryId>
  <url>https://example.com/dm02</url><price>1000</price>
  <description><![CDATA[<p>Для измерения избыточного давления</p>]]></description>
  <param name="Артикул">A-1</param><param name="Класс точности">1,5</param>
  <param name="Диаметр корпуса" unit="мм">100</param>
</offer></offers></shop></yml_catalog>"""


def write(tmp_path, text: str) -> str:
    path = tmp_path / "feed.xml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_real_feed_excerpt_with_browser_garbage():
    # Фрагмент реального фида, скопированный из браузера: текст перед XML и <script> от расширений.
    root = load_feed(str(FIXTURE))
    stats = audit(root)
    assert stats["offers"] == 20
    assert stats["with_description"] == 0 and stats["with_specs"] == 0

    [product] = find_offers(root, ["58726"])
    assert product["name"] == "БП15Б-Д2-24"
    assert product["manufacturer"] == "ОВЕН"
    assert product["category"].endswith("Промышленные блоки питания на DIN-рейку")
    assert product["identifiers"]["Артикул"] == "96440"
    assert product["specs"] == {}  # «Артикул» и «Код товара» — не характеристики
    assert "price" not in product


def test_offer_with_params_and_html_description(tmp_path):
    [product] = find_offers(load_feed(write(tmp_path, FEED_WITH_PARAMS)), ["7"])
    assert product["category"] == "КИП / Манометры"
    assert product["specs"] == {"Класс точности": "1,5", "Диаметр корпуса": "100 мм"}
    assert product["identifiers"]["Артикул"] == "A-1"
    result = parse_input(product)  # импорт даёт корректный вход для генератора
    assert result.product.description == "Для измерения избыточного давления"
    assert result.profile == "manometer"


def test_missing_offer_id(tmp_path):
    with pytest.raises(FeedError, match="нет товаров с id: 999"):
        find_offers(load_feed(write(tmp_path, FEED_WITH_PARAMS)), ["999"])


@pytest.mark.parametrize("text, expected", [
    ("<html><body/></html>", "не YML"),
    ("<yml_catalog><shop>", "некорректный XML"),
    ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><yml_catalog><shop>&a;</shop></yml_catalog>',
     "запрещённые"),
])
def test_bad_feeds(tmp_path, text, expected):
    with pytest.raises(FeedError, match=expected):
        load_feed(write(tmp_path, text))
