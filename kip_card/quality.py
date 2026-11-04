"""Проверки качества текста, не связанные с фактами (факты — в factcheck.py).

Появились после первого прогона на реальной модели (docs/DECISIONS.md, раздел 10):
- заголовок со строчной буквы и с «;» — исправляем сами, это безопасно;
- запрос, не относящийся к товару (модель скопировала пример из промпта) — переспрашиваем;
- оценочные слова («идеальный», «надёжный») без опоры на данные — переспрашиваем.

Второй прогон добавил:
- повтор производителя в заголовке («testo 830-T1 Testo») — убираем сами;
- «Кому подходит» из одних должностей («технические специалисты», «снабженцы») — переспрашиваем.

Сравнение моделей (gpt-5.4-mini, Qwen) добавило:
- бренд и модель строчными, когда модель вставляет в текст поисковый запрос
  («овен пд100-…») — возвращаем написание из данных сами;
- служебные обороты о данных вместо текста о приборе («по данным карточки»,
  «этот прибор ищут как») — переспрашиваем;
- запросы с многоточиями и знаками градуса — переспрашиваем;
- характеристики и числа в «Кому подходит» — переспрашиваем.
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
    Модель из коротких частей («dm 02» при названии «… DM 02») узнаётся целиком.
    """
    identity = _identity(product)
    identity_compact = re.sub(r"[\s\-]", "", identity)
    bad = []
    for query in queries:
        words = [w.strip(".,/") for w in _WORD_RE.findall(_plain(query))]
        content = re.sub(r"[\s\-]", "", " ".join(w for w in words if w and w not in _QUERY_STOPWORDS))
        relevant = len(content) >= 3 and content in identity_compact
        for word in ([] if relevant else _WORD_RE.findall(_plain(query))):
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


# ------------------------------------------------------------------ написание бренда и модели


def _case_map(product: ProductInput) -> dict[str, str]:
    """Строчная форма → написание из данных.

    Производитель — всегда (это бренд). Из названия — только слова с цифрами или целиком
    заглавные (ПД100-ДИ1,6-111-0,5, DM): обычные слова вроде «Манометр» не трогаем.
    Если в названии бренд пишется строчными (testo), побеждает написание из названия.
    """
    mapping: dict[str, str] = {}
    for token in product.manufacturer.split():
        token = token.strip(".,;:()«»\"")
        if len(token) >= 2:
            mapping[token.lower()] = token
    for token in product.name.split():
        token = token.strip(".,;:()«»\"")
        if len(token) < 2:
            continue
        if token == token.lower():
            mapping[token] = token  # написание производителя строчными — сохранить
        elif any(ch.isdigit() for ch in token) or token == token.upper():
            mapping[token.lower()] = token
    return {low: canon for low, canon in mapping.items() if low != canon}


def restore_case(text: str, product: ProductInput) -> str:
    """«датчик давления овен пд100-ди1,6-111-0,5» → «датчик давления ОВЕН ПД100-ДИ1,6-111-0,5».

    Заменяются только вхождения, написанные целиком строчными.
    """
    for low, canon in _case_map(product).items():
        pattern = re.compile(rf"(?<![\w-]){re.escape(low)}(?![\w-])", re.IGNORECASE)
        text = pattern.sub(lambda m: canon if m.group() == m.group().lower() else m.group(), text)
    return text


# ------------------------------------------------------------------ служебные обороты

_SERVICE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bпо данным\b",
        r"\bпо описанию\b",
        r"\bв описании\b",
        r"\bв карточк\w*",
        r"\bвходн\w* данн\w*",
        r"\bищут как\b",
        r"\bв (?:поисковых )?запросах\b",
        r"\bуказанн\w* характеристик\w*",
        r"\bхарактеристики\b[^.]{0,60}?\bвключа\w*",
        r"\bне (?:указан|приведен)\w*",
    )
]


def service_phrases(text: str) -> list[str]:
    """Обороты о данных и карточке вместо текста о приборе («по данным карточки…»)."""
    plain = text.replace("ё", "е")
    found = []
    for pattern in _SERVICE_PATTERNS:
        match = pattern.search(plain)
        if match:
            found.append(match.group())
    return found


# ------------------------------------------------------------------ запросы и «Кому подходит»

_QUERY_SYMBOLS = ("…", "...", "°", "±")


def unnatural_queries(queries: list[str]) -> list[str]:
    """Запросы с многоточием, знаком градуса или «±» — так в поиске не набирают."""
    return [q for q in queries if any(sym in q for sym in _QUERY_SYMBOLS)]


def specs_in_audience(who_for: list[str]) -> list[str]:
    """Пункты «Кому подходит» с числами — это характеристики, а не места применения."""
    return [item for item in who_for if re.search(r"\d", item)]


# ------------------------------------------------------------------ повторы и сокращения

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])")
_STEM_WORD_RE = re.compile(r"[а-яёa-z]{4,}", re.IGNORECASE)


def _stems(sentence: str) -> set[str]:
    return {w.lower().replace("ё", "е")[:5] for w in _STEM_WORD_RE.findall(sentence)}


def repeated_sentences(text: str, threshold: float = 0.7, min_words: int = 4) -> list[str]:
    """Предложения, которые почти дословно повторяют смысл предыдущих (те же основы слов).

    Мера — доля общих основ от меньшего предложения. Реальный пример: «Показывающий манометр…
    предназначен для измерения избыточного давления» и в конце «Прибор относится к показывающим
    манометрам и предназначен для измерения избыточного давления».
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(text.replace("\n", " ")) if s.strip()]
    seen: list[set[str]] = []
    repeats = []
    for sentence in sentences:
        stems = _stems(sentence)
        if len(stems) >= min_words:
            for prev in seen:
                if len(prev) >= min_words and len(stems & prev) / min(len(stems), len(prev)) >= threshold:
                    repeats.append(sentence)
                    break
        seen.append(stems)
    return repeats


_ABBREV_RE = re.compile(r"(?<![\w.])[а-яё]{1,5}\.\s?[а-яё]{1,6}\.(?![\w])", re.IGNORECASE)


def abbreviations(text: str) -> list[str]:
    """Нераскрытые сокращения из данных вроде «изм. знач.» в тексте для покупателя."""
    return list(dict.fromkeys(m.group() for m in _ABBREV_RE.finditer(text)))
