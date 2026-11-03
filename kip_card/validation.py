"""Разбор и проверка входных данных. Работает без LLM и полностью детерминирован.

- некорректный JSON и непригодная структура → InputError с понятным текстом;
- пустые поля и пропущенные характеристики → замечания (Issue), генерация продолжается;
- для типа прибора проверяется набор ключевых характеристик, которых ждёт покупатель.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field

from .models import Issue, ProductInput

MAX_INPUT_CHARS = 60_000
MAX_DESCRIPTION_CHARS = 8_000
SHORT_DESCRIPTION_CHARS = 150
# «Мало данных»: короткое описание и немного характеристик — честный короткий текст
# важнее требования «800 символов» (см. docs/DECISIONS.md, раздел 10)
SPARSE_MAX_SPECS = 8

ALIASES = {
    "name": ("name", "title", "наименование", "название"),
    "manufacturer": ("manufacturer", "vendor", "brand", "производитель"),
    "product_type": ("product_type", "type", "typePrefix", "тип"),
    "category": ("category", "категория"),
    "specs": ("specs", "characteristics", "params", "attributes", "характеристики"),
    "description": ("description", "описание"),
    "identifiers": ("identifiers", "ids"),
    "source_url": ("source_url", "url"),
}


class InputError(Exception):
    """Входные данные нельзя обработать. issues объясняют почему."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(i.message for i in issues))


@dataclass
class ValidationResult:
    product: ProductInput
    issues: list[Issue] = field(default_factory=list)
    missing_specs: list[str] = field(default_factory=list)  # ключевые характеристики, которых нет
    empty_specs: list[str] = field(default_factory=list)  # характеристики, переданные без значения
    profile: str = "generic"

    @property
    def sparse(self) -> bool:
        """Данных слишком мало, чтобы написать 800+ символов без выдумок."""
        return (len(self.product.description) < SHORT_DESCRIPTION_CHARS
                and len(self.product.specs) < SPARSE_MAX_SPECS)


# ---------------------------------------------------------------- профили приборов
# Ключевые характеристики, которые покупатель ожидает увидеть для типа прибора.
# Каждая — (подпись в карточке, [фрагменты названия характеристики во входных данных]).
# Порядок важен: профиль выбирается по первому совпадению.
PROFILES: dict[str, dict] = {
    "pressure_sensor": {
        "title": "датчик давления",
        "detect": ("датчик давления", "преобразователь давления", "датчик избыточного",
                   "датчик абсолютного", "датчик дифференциального", "датчик перепада"),
        "keys": [
            ("Диапазон измерений", ["диапазон измер", "предел измер", "впи"]),
            ("Точность", ["класс точности", "погрешност", "точност"]),
            ("Выходной сигнал", ["выходной сигнал", "выход", "интерфейс"]),
            ("Напряжение питания", ["питани"]),
            ("Присоединение к процессу", ["присоедин", "штуцер", "резьб"]),
            ("Степень защиты", ["степень защиты", "ip"]),
        ],
    },
    "manometer": {
        "title": "манометр",
        "detect": ("манометр", "вакуумметр", "мановакуумметр", "напоромер", "тягомер"),
        "keys": [
            ("Диапазон измерений", ["диапазон", "предел", "впи"]),
            ("Класс точности", ["класс точности", "точност", "погрешност"]),
            ("Присоединение", ["присоедин", "штуцер", "резьб"]),
            ("Диаметр корпуса", ["диаметр"]),
            ("Степень защиты", ["степень защиты", "ip"]),
        ],
    },
    "pyrometer": {
        "title": "пирометр",
        "detect": ("пирометр", "инфракрасный термометр", "ик-термометр", "ик термометр"),
        "keys": [
            ("Диапазон измерений", ["диапазон"]),
            ("Погрешность", ["погрешност", "точност"]),
            ("Оптическое разрешение", ["оптическ", "визирован", "фокус"]),
            ("Питание", ["питани", "батаре", "аккумулятор"]),
        ],
    },
    "regulator": {
        "title": "измеритель-регулятор",
        "detect": ("регулятор", "измеритель-регулятор", "контроллер температуры", "терморегулятор"),
        "keys": [
            ("Входы (типы датчиков)", ["вход", "датчик"]),
            ("Выходы", ["выход"]),
            ("Напряжение питания", ["питани"]),
            ("Интерфейс связи", ["интерфейс", "rs-485", "rs485", "modbus"]),
            ("Корпус / монтаж", ["корпус", "монтаж", "щит", "din"]),
        ],
    },
    "generic": {
        "title": "прибор",
        "detect": (),
        # Набор из ТЗ: диапазон, класс точности, питание, тип корпуса, интерфейсы.
        "keys": [
            ("Диапазон измерений", ["диапазон", "предел"]),
            ("Точность", ["класс точности", "погрешност", "точност"]),
            ("Питание", ["питани", "батаре", "аккумулятор"]),
            ("Корпус / степень защиты", ["корпус", "степень защиты", "ip"]),
            ("Интерфейсы", ["интерфейс", "выходной сигнал", "rs-485", "usb", "bluetooth"]),
        ],
    },
}


def detect_profile(product: ProductInput) -> str:
    text = " ".join([product.product_type, product.name, product.category]).lower()
    for key, profile in PROFILES.items():
        if any(word in text for word in profile["detect"]):
            return key
    return "generic"


def _has_spec(spec_names: list[str], fragments: list[str]) -> bool:
    for name in spec_names:
        for frag in fragments:
            if frag == "ip":
                if re.search(r"\bip\b", name) or name.startswith("ip"):
                    return True
            elif frag in name:
                return True
    return False


# ---------------------------------------------------------------- нормализация
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v\u00a0\u202f]+")


def clean_text(value: str) -> str:
    """Убирает HTML-теги (описания из CMS часто приходят в HTML) и лишние пробелы."""
    text = html.unescape(_TAG_RE.sub(" ", value))
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return "; ".join(t for t in (_to_text(v) for v in value) if t)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {t}" for k, v in value.items() if (t := _to_text(v)))
    return clean_text(str(value))


def _pick(data: dict, field_name: str):
    for key in ALIASES[field_name]:
        if key in data:
            return key, data[key]
    return None, None


def _parse_specs(raw, issues: list[Issue]) -> tuple[dict[str, str], list[str]]:
    specs: dict[str, str] = {}
    empty: list[str] = []
    if raw is None:
        return specs, empty
    if isinstance(raw, list):  # [{"name": ..., "value": ...}] — частый формат выгрузок
        pairs = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                pairs.append((item.get("name"), item.get("value")))
            else:
                raise InputError([Issue(level="error", field="specs",
                                        message="specs: элементы списка должны иметь вид "
                                                "{\"name\": ..., \"value\": ...}")])
    elif isinstance(raw, dict):
        pairs = list(raw.items())
    else:
        raise InputError([Issue(level="error", field="specs",
                                message="specs должно быть объектом {\"характеристика\": \"значение\"}")])

    for key, value in pairs:
        name = _to_text(key)
        if not name:
            continue
        text = _to_text(value)
        if text:
            specs[name] = text
        else:
            empty.append(name)
            issues.append(Issue(level="warning", field=f"specs.{name}",
                                message=f"Характеристика «{name}» передана без значения"))
    return specs, empty


# ---------------------------------------------------------------- основной разбор
def parse_input(raw: str | bytes | dict) -> ValidationResult:
    """Принимает JSON-строку или dict, возвращает нормализованный товар и замечания."""
    if isinstance(raw, (str, bytes)):
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        if not text.strip():
            raise InputError([Issue(level="error", field="input", message="Пустой ввод: передайте JSON с данными о товаре")])
        if len(text) > MAX_INPUT_CHARS:
            raise InputError([Issue(level="error", field="input",
                                    message=f"Слишком большой ввод ({len(text)} символов, максимум {MAX_INPUT_CHARS})")])
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError([Issue(level="error", field="input",
                                    message=f"Некорректный JSON: {exc.msg} (строка {exc.lineno}, столбец {exc.colno})")]) from exc
    else:
        data = raw

    if not isinstance(data, dict):
        raise InputError([Issue(level="error", field="input",
                                message="Ожидается JSON-объект с полями name, manufacturer, specs, description")])

    issues: list[Issue] = []
    known_keys: set[str] = set()
    values: dict = {}
    for field_name in ALIASES:
        key, value = _pick(data, field_name)
        if key is not None:
            known_keys.add(key)
        values[field_name] = value

    name = _to_text(values["name"])
    if not name:
        raise InputError([Issue(level="error", field="name",
                                message="Не указано название товара (name) — без него карточку не собрать")])

    specs, empty_specs = _parse_specs(values["specs"], issues)

    identifiers_raw = values["identifiers"]
    identifiers = ({_to_text(k): _to_text(v) for k, v in identifiers_raw.items() if _to_text(v)}
                   if isinstance(identifiers_raw, dict) else {})

    description = _to_text(values["description"])
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0]
        issues.append(Issue(level="info", field="description",
                            message=f"Описание сокращено до {MAX_DESCRIPTION_CHARS} символов"))

    product = ProductInput(
        name=name,
        manufacturer=_to_text(values["manufacturer"]),
        product_type=_to_text(values["product_type"]),
        category=_to_text(values["category"]),
        specs=specs,
        description=description,
        identifiers=identifiers,
        source_url=_to_text(values["source_url"]),
    )

    unknown = [k for k in data if k not in known_keys]
    if unknown:
        issues.append(Issue(level="info", field="input",
                            message="Неизвестные поля проигнорированы: " + ", ".join(unknown)))

    return _check_completeness(product, issues, empty_specs)


def _check_completeness(product: ProductInput, issues: list[Issue], empty_specs: list[str]) -> ValidationResult:
    if not product.manufacturer:
        issues.append(Issue(level="warning", field="manufacturer", message="Не указан производитель"))
    if not product.specs:
        issues.append(Issue(level="warning", field="specs",
                            message="Нет технических характеристик — в карточке будет «данных нет», "
                                    "текст получится общим"))
    if not product.description:
        issues.append(Issue(level="warning", field="description",
                            message="Нет описания — назначение прибора модель будет знать только из названия"))
    else:
        if len(product.description) < SHORT_DESCRIPTION_CHARS:
            issues.append(Issue(level="warning", field="description",
                                message=f"Короткое описание ({len(product.description)} симв.) — "
                                        "добавьте фрагмент паспорта, иначе SEO-текст будет общим"))
        if product.description.rstrip().endswith(("...", "…")):
            issues.append(Issue(level="warning", field="description",
                                message="Описание похоже на обрезанное (заканчивается многоточием)"))

    profile_key = detect_profile(product)
    spec_names = [n.lower() for n in product.specs]
    missing: list[str] = []
    for label, fragments in PROFILES[profile_key]["keys"]:
        if not _has_spec(spec_names, fragments):
            missing.append(label)
    if missing:
        title = PROFILES[profile_key]["title"]
        issues.append(Issue(level="warning", field="specs",
                            message=f"Для типа «{title}» не хватает ключевых характеристик: "
                                    + ", ".join(missing)))

    return ValidationResult(product=product, issues=issues, missing_specs=missing,
                            empty_specs=empty_specs, profile=profile_key)
