"""Проверка сгенерированного текста на выдуманные технические значения.

Идея: любое число или технический код (IP65, RS-485, G1/2, AISI 316L) в тексте модели
должен встречаться во входных данных. Сравнение нормализованное: «1,6» = «1.6»,
«1,0» = «1», «500 000» = «500000». Всё, что не нашлось, — повод переспросить модель,
а если не помогло — предупреждение контент-менеджеру.

Ограничение (честно): единицы измерения не сверяются. «1,6 бар» вместо «1,6 МПа» эта
проверка не поймает — см. docs/DECISIONS.md.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# Число: 1 / 1,6 / 0.5 / 500 000 (группы разрядов через пробел или неразрывный пробел)
_NUMBER_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?![\d,.]\d)|\d+(?:[.,]\d+)?")
# Технический код: латиница + цифры (IP65, RS-485, G1/2, M20x1,5, AISI 316L, EN175301-803)
_CODE_RE = re.compile(r"\b[A-Za-z]{1,12}[-\s]?\d[\w\-/.,]*", re.UNICODE)
# Аббревиатуры латиницей (HART, ABS, NTC, USB) — тоже технические факты
_ABBR_RE = re.compile(r"\b[A-Z]{2,}\b")
# Любое латинское слово (Bluetooth, Modbus, Wi-Fi): в русском тексте это почти всегда
# название технологии или бренда — тоже должно быть во входных данных
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}(?:-[A-Za-z]+)*\b")
# Общеупотребительные латинские сокращения, которые не являются характеристиками
_ABBR_ALLOWED = {"SEO", "B2B"}


def _canon_number(token: str) -> str | None:
    token = re.sub(r"[ \u00a0\u202f]", "", token).replace(",", ".")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return format(value.normalize(), "f")


def numbers_in(text: str) -> set[str]:
    found: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        canon = _canon_number(match.group())
        if canon is not None:
            found.add(canon)
    return found


# Кириллица, похожая на латиницу: на сайтах «М20х1,5» часто набирают русскими буквами.
_HOMOGLYPHS = str.maketrans("авекмнорстхуё", "abekmhopctxye")


def _compact(text: str) -> str:
    return re.sub(r"[\s\-]", "", text).lower().translate(_HOMOGLYPHS)


def _canon_code(token: str) -> str:
    return _compact(token).rstrip(".,")


class FactChecker:
    """Знает все значения из входных данных и ищет в тексте те, которых там нет."""

    def __init__(self, source_texts: list[str]):
        source = "\n".join(t for t in source_texts if t)
        self._numbers = numbers_in(source)
        self._source_compact = _compact(source)

    def check(self, text: str) -> list[str]:
        """Список значений из text, которых нет во входных данных.

        Сначала технические коды (RS-485, IP67), затем отдельные числа. Число внутри
        уже найденного кода второй раз не показываем, чтобы не плодить шум.
        """
        found: list[str] = []
        code_spans: list[tuple[int, int]] = []
        for match in _CODE_RE.finditer(text):
            code_spans.append(match.span())
            if match.group().rstrip(".,") in _ABBR_ALLOWED:
                continue
            if _canon_code(match.group()) not in self._source_compact:
                found.append(match.group().strip().rstrip(".,"))

        def inside_code(span: tuple[int, int]) -> bool:
            return any(s <= span[0] and span[1] <= e for s, e in code_spans)

        for match in list(_ABBR_RE.finditer(text)) + list(_LATIN_WORD_RE.finditer(text)):
            raw = match.group()
            if raw in _ABBR_ALLOWED or inside_code(match.span()):
                continue
            if _canon_code(raw) not in self._source_compact:
                found.append(raw)

        for match in _NUMBER_RE.finditer(text):
            if inside_code(match.span()):
                continue
            canon = _canon_number(match.group())
            if canon is not None and canon not in self._numbers:
                found.append(match.group().strip())
        return _unique(found)


def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
