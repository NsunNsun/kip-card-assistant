"""Импорт товаров из YML-фида (формат Яндекс.Маркета, который выгружает 1С-Битрикс).

YML даёт название, тип, производителя, категорию и ссылку. Технические характеристики
и описание в фиде «Союз-Прибора» не выгружаются, поэтому импорт создаёт заготовку входного
JSON: контент-менеджер дополняет specs и description (или сервис сгенерирует карточку
с пометками «данных нет»). Цены, остатки и картинки в карточку не передаются.
"""

from __future__ import annotations

import urllib.request
from collections import Counter
from pathlib import Path
from xml.etree.ElementTree import Element

from defusedxml import DefusedXmlException
from defusedxml import ElementTree as SafeET  # защита от XXE и «billion laughs»

# Параметры фида, которые являются идентификаторами, а не характеристиками
IDENTIFIER_PARAMS = {"Артикул", "Код товара"}


class FeedError(Exception):
    pass


def load_feed(source: str, timeout: float = 60.0) -> Element:
    """Читает фид из файла или по URL."""
    try:
        if source.startswith(("http://", "https://")):
            request = urllib.request.Request(source, headers={"User-Agent": "kip-card-assistant/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 — URL задаёт пользователь
                data = resp.read()
        else:
            data = Path(source).read_bytes()
    except OSError as exc:
        raise FeedError(f"Не удалось прочитать фид {source}: {exc}") from exc

    start = data.find(b"<")  # отрезаем мусор перед XML (например, при копировании из браузера)
    if start == -1:
        raise FeedError("В файле нет XML")
    try:
        root = SafeET.fromstring(data[start:])
    except SafeET.ParseError as exc:
        raise FeedError(f"Фид — некорректный XML: {exc}") from exc
    except DefusedXmlException as exc:
        raise FeedError(f"Фид содержит запрещённые XML-конструкции (сущности/DTD): {exc}") from exc
    if root.find("shop") is None:
        raise FeedError("Это не YML-фид: нет элемента <shop>")
    return root


def _text(el: Element | None, tag: str) -> str:
    child = el.find(tag) if el is not None else None
    return (child.text or "").strip() if child is not None and child.text else ""


def category_paths(root: Element) -> dict[str, str]:
    raw = {}
    for cat in root.iter("category"):
        raw[cat.get("id", "")] = (cat.text or "").strip(), cat.get("parentId")
    paths: dict[str, str] = {}
    for cat_id in raw:
        names, current, seen = [], cat_id, set()
        while current and current in raw and current not in seen:
            seen.add(current)
            name, parent = raw[current]
            names.append(name)
            current = parent
        paths[cat_id] = " / ".join(reversed(names))
    return paths


def offer_to_input(offer: Element, categories: dict[str, str]) -> dict:
    """Превращает <offer> во входной JSON сервиса."""
    specs: dict[str, str] = {}
    identifiers: dict[str, str] = {"id в YML-фиде": offer.get("id", "")}
    for param in offer.findall("param"):
        name = (param.get("name") or "").strip()
        value = (param.text or "").strip()
        unit = (param.get("unit") or "").strip()
        if not name or not value:
            continue
        if name in IDENTIFIER_PARAMS:
            identifiers[name] = value
        else:
            specs[name] = f"{value} {unit}".strip()
    if _text(offer, "vendorCode") and "Артикул" not in identifiers:
        identifiers["Артикул"] = _text(offer, "vendorCode")

    name = _text(offer, "model") or _text(offer, "name")
    return {
        "name": name,
        "product_type": _text(offer, "typePrefix"),
        "manufacturer": _text(offer, "vendor"),
        "category": categories.get(_text(offer, "categoryId"), ""),
        "specs": specs,
        "description": _text(offer, "description"),
        "identifiers": {k: v for k, v in identifiers.items() if v},
        "source_url": _text(offer, "url"),
    }


def find_offers(root: Element, offer_ids: list[str]) -> list[dict]:
    categories = category_paths(root)
    wanted = set(offer_ids)
    found = {o.get("id"): offer_to_input(o, categories) for o in root.iter("offer") if o.get("id") in wanted}
    missing = [i for i in offer_ids if i not in found]
    if missing:
        raise FeedError("В фиде нет товаров с id: " + ", ".join(missing))
    return [found[i] for i in offer_ids]


def audit(root: Element) -> dict:
    """Сводка: насколько данных фида хватает для карточек товаров."""
    categories = category_paths(root)
    total = with_desc = with_specs = 0
    by_category: Counter[str] = Counter()
    for offer in root.iter("offer"):
        total += 1
        has_desc = bool(_text(offer, "description"))
        has_specs = any((p.get("name") or "") not in IDENTIFIER_PARAMS for p in offer.findall("param"))
        with_desc += has_desc
        with_specs += has_specs
        if not has_desc and not has_specs:
            by_category[categories.get(_text(offer, "categoryId"), "без категории")] += 1
    return {
        "offers": total,
        "with_description": with_desc,
        "with_specs": with_specs,
        "without_both_by_category": by_category.most_common(),
    }
