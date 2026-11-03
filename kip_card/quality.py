"""Проверки качества текста, не связанные с фактами (факты — в factcheck.py).

Появились после первого прогона на реальной модели (docs/DECISIONS.md, раздел 10):
- заголовок со строчной буквы и с «;» — исправляем сами, это безопасно;
- запрос, не относящийся к товару (модель скопировала пример из промпта) — переспрашиваем;
- оценочные слова («идеальный», «надёжный») без опоры на данные — переспрашиваем.

Второй прогон добавил:
- повтор производителя в заголовке («testo 830-T1 Testo») — убираем сами;
- «Кому подходит» из одних должностей («технические специалисты», «снабженцы») — переспрашиваем.
"""

from __future__ import annotations

import re

from .models import ProductInput

# Слова, не делающие запрос «про этот товар»
_QUERY_STOPWORDS = {
    "купить", "цена", "цены", "стоимость", "заказать", "недорого", "доставка", "москва",
    "для", "под", "при", "или", "как", "что", "это", "интернет", "магазин", "отзывы",
    "характеристики", "описание", "инструкция",
}

# Оценочные слова: допустимы, только если они есть во входных данных
_EVALUATIVE = [
    (re.compile(r"\bлучш\w*", re.I), "лучший"),
    (re.compile(r"\bсам(ый|ая|ое|ые|ого|ой|ым|ых)\b", re.I), "самый"),
    (re.compile(r"№\s*1\b"), "№1"),
    (re.compile(r"\bуникальн\w*", re.I), "уникальный"),
    (re.compile(r"\bинновацион\w*", re.I), "инновационный"),
    (re.compile(r"\bидеальн\w*", re.I), "идеальный"),
    (re.compile(r"\bнад[её]жн\w*", re.I), "надёжный"),
    (re.compile(r"\bвысококачествен\w*", re.I), "высококачественный"),
    (re.compile(r"\bпревосходн\w*", re.I), "превосходный"),
    (re.compile(r"\bнепревзойд\w*", re.I), "непревзойдённый"),
]

_WORD_RE = re.compile(r"[\w][\w\-./,]*", re.UNICODE)

# Слова-должности: пункт «Кому подходит» только из них ничего не говорит о применении прибора
_ROLE_STEMS = ("специалист", "снабжен", "инженер", "техническ", "покупател", "сотрудник",
               "кипиа", "менеджер", "закупщик", "работник", "персонал", "профессионал")
_ROLE_FILLER = {"и", "по", "в", "для", "отдел", "отделы", "служб", "службы", "предприятий", "области"}


def _plain(text: str) -> str:
    return text.lower().replace("ё", "е")


def _drop_repeated_manufacturer(title: str, product: ProductInput) -> str:
    """«Пирометр testo 830-T1 Testo» → «Пирометр testo 830-T1», если производитель уже в названии."""
    maker = product.manufacturer.strip()
    if not maker or maker.lower() not in product.name.lower() or product.name not in title:
        return title
    before, after = title.split(product.name, 1)
    pattern = re.compile(rf"(?<!\w){re.escape(maker)}(?!\w)", re.I)
    before, after = pattern.sub("", before), pattern.sub("", after)
    result = f"{before}{product.name}{after}"
    result = re.sub(r"\s+,", ",", re.sub(r"\s{2,}", " ", result))
    return re.sub(r",\s*,", ",", result).strip(" ,")


def fix_title(title: str, product: ProductInput) -> str:
    """Запятая вместо «;», без повтора производителя, заглавная первая буква.

    Название модели со строчной по написанию производителя (testo) не трогаем.
    """
    title = re.sub(r"\s*;\s*", ", ", title).strip(" ,")
    title = _drop_repeated_manufacturer(title, product)
    if not title:
        return title
    first_word = title.split()[0]
    if first_word in product.name.split() or first_word in product.manufacturer.split():
        return title  # написание производителя: «testo 830-T1»
    return title[0].upper() + title[1:]


def _identity(product: ProductInput) -> str:
    return _plain(" ".join([product.name, product.product_type, product.category, product.manufacturer]))


def irrelevant_queries(queries: list[str], product: ProductInput) -> list[str]:
    """Запросы, в которых нет ни одного слова из названия, типа, категории или производителя.

    Сравнение по основе слова (первые 5 букв), чтобы «давления» совпадало с «давление».
    """
    identity = _identity(product)
    identity_compact = re.sub(r"[\s\-]", "", identity)
    bad = []
    for query in queries:
        relevant = False
        for word in _WORD_RE.findall(_plain(query)):
            word = word.strip(".,/")
            if len(word) < 3 or word in _QUERY_STOPWORDS:
                continue
            if word.isalpha():
                stem = word[:5]
                if stem in identity:
                    relevant = True
                    break
            elif re.sub(r"[\-]", "", word) in identity_compact:  # модель: «пд100-ди1,6-111-0,5»
                relevant = True
                break
        if not relevant:
            bad.append(query)
    return bad


def evaluative_words(text: str, source: str) -> list[str]:
    """Оценочные слова в тексте, которых нет во входных данных."""
    source_plain = _plain(source)
    found = []
    for pattern, label in _EVALUATIVE:
        match = pattern.search(text)
        if match and not pattern.search(source_plain):
            found.append(match.group())
    return found


def generic_audience(who_for: list[str]) -> list[str]:
    """Пункты «Кому подходит», состоящие только из должностей, без объектов и сфер применения."""
    bad = []
    for item in who_for:
        words = [w for w in re.findall(r"[а-яёa-z]+", _plain(item)) if w not in _ROLE_FILLER]
        if words and all(any(w.startswith(stem) for stem in _ROLE_STEMS) for w in words):
            bad.append(item)
    return bad
