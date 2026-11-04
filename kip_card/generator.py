"""Сборка карточки: проверка входа → модель → проверки ответа → (исправление) → карточка.

Проверки ответа: детерминированные (длины, фактчекер, качество) и, если включена,
смысловая — вторым запросом к модели (judge.py). Замечания обеих идут в один цикл исправления.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from .factcheck import FactChecker
from .judge import Finding, Judge, field_texts, quote_found
from .llm import JSONModel, LLMError, LLMResponseError
from .models import NO_DATA, Issue, LLMCopy, Meta, ProductCard, ProductInput, SpecRow
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .quality import (abbreviations, evaluative_words, fix_title, generic_audience, irrelevant_queries,
                      repeated_sentences, restore_case, service_phrases, specs_in_audience, unnatural_queries)
from .validation import ValidationResult, parse_input

log = logging.getLogger(__name__)

TITLE_MAX = 70
DESCRIPTION_MIN, DESCRIPTION_MAX = 800, 1200
QUERIES_MIN, QUERIES_MAX = 3, 5


def _source_texts(product: ProductInput) -> list[str]:
    texts = [product.name, product.product_type, product.manufacturer, product.category, product.description]
    for key, value in product.specs.items():
        texts += [key, value]
    return texts


def _normalize(copy: LLMCopy, product: ProductInput) -> LLMCopy:
    queries: list[str] = []
    for q in copy.search_queries:
        q = " ".join(q.split()).strip(" .").lower()  # запросы набирают строчными
        if q and q.lower() not in {x.lower() for x in queries}:
            queries.append(q)
    fix = lambda text: restore_case(text, product)  # noqa: E731
    commercial = copy.commercial.model_copy(update={
        "summary": fix(copy.commercial.summary),
        "who_for": [fix(x) for x in copy.commercial.who_for],
        "tasks": [fix(x) for x in copy.commercial.tasks],
    })
    return copy.model_copy(update={
        "seo_title": fix_title(fix(" ".join(copy.seo_title.split()).strip(" .\"«»")), product),
        "seo_description": fix("\n\n".join(p.strip() for p in copy.seo_description.strip().split("\n\n") if p.strip())),
        "search_queries": queries,  # запросы остаются строчными
        "commercial": commercial,
    })


def review(copy: LLMCopy, checker: FactChecker, product: ProductInput, sparse: bool = False) -> list[str]:
    """Детерминированные проверки ответа модели. Пустой список — всё в порядке.

    sparse — во входе мало данных: нижняя граница длины описания не требуется,
    чтобы модель не добивала текст выдумками (короткий текст помечается предупреждением).
    """
    problems: list[str] = []
    title_len = len(copy.seo_title)
    if title_len > TITLE_MAX:
        problems.append(f"seo_title: {title_len} символов, нужно не больше {TITLE_MAX}")
    if not copy.seo_title:
        problems.append("seo_title пустой")
    desc_len = len(copy.seo_description)
    if desc_len > DESCRIPTION_MAX or (desc_len < DESCRIPTION_MIN and not sparse):
        problems.append(f"seo_description: {desc_len} символов, нужно от {DESCRIPTION_MIN} до {DESCRIPTION_MAX}")
    if not QUERIES_MIN <= len(copy.search_queries) <= QUERIES_MAX:
        problems.append(f"search_queries: {len(copy.search_queries)} шт., нужно от {QUERIES_MIN} до {QUERIES_MAX}")
    if not copy.commercial.tasks:
        problems.append("commercial: заполни tasks")
    if not copy.commercial.who_for and not sparse:
        problems.append("commercial: заполни who_for — места применения из описания")
    repeats = repeated_sentences(copy.seo_description)
    if repeats:
        problems.append("seo_description: повтор уже сказанного — " + "; ".join(f"«{x}»" for x in repeats)
                        + ". Убери повтор или замени новым фактом из данных")
    short = abbreviations("\n".join([copy.seo_description, copy.commercial.summary, *copy.commercial.tasks]))
    if short:
        problems.append("seo_description: нераскрытые сокращения — " + ", ".join(f"«{x}»" for x in short)
                        + ". Напиши слова полностью, числа не меняй")
    numbered = specs_in_audience(copy.commercial.who_for)
    if numbered:
        problems.append("commercial.who_for: характеристики вместо мест применения — "
                        + "; ".join(f"«{x}»" for x in numbered) + ". Укажи объекты и системы, где применяется прибор, без чисел")
    roles = generic_audience(copy.commercial.who_for)
    if roles:
        problems.append("commercial.who_for: пункты из одних должностей, без объектов и сфер применения — "
                        + "; ".join(f"«{r}»" for r in roles) + ". Укажи, где применяется этот прибор")

    fields = {
        "seo_title": copy.seo_title,
        "seo_description": copy.seo_description,
        "search_queries": "\n".join(copy.search_queries),
        "commercial": "\n".join([copy.commercial.summary, *copy.commercial.who_for, *copy.commercial.tasks]),
    }
    source = "\n".join(_source_texts(product))
    for name, text in fields.items():
        unknown = checker.check(text)
        if unknown:
            problems.append(f"{name}: значения, которых нет во входных данных — " + ", ".join(unknown))
        if name in ("seo_description", "commercial"):
            service = service_phrases(text)
            if service:
                problems.append(f"{name}: служебные обороты о данных вместо текста о приборе — "
                                + ", ".join(f"«{x}»" for x in service)
                                + ". Перепиши фразу о самом приборе; чего нет в данных — укажи в data_gaps")
        if name != "search_queries":
            words = evaluative_words(text, source)
            if words:
                problems.append(f"{name}: оценочные слова без опоры на данные — " + ", ".join(words)
                                + ". Убери их или замени фактом из данных")

    odd = unnatural_queries(copy.search_queries)
    if odd:
        problems.append("search_queries: так в поиске не набирают (многоточие, градус, «±») — "
                        + "; ".join(f"«{q}»" for q in odd))
    unrelated = irrelevant_queries(copy.search_queries, product)
    if unrelated:
        problems.append("search_queries: запросы не про этот товар (нет модели, типа прибора или "
                        "производителя) — " + "; ".join(f"«{q}»" for q in unrelated))
    return problems


def _truncate_title(title: str) -> str:
    if len(title) <= TITLE_MAX:
        return title
    cut = title[:TITLE_MAX].rsplit(" ", 1)[0]
    return cut.rstrip(" ,–-:")


def _dedupe(names: list[str]) -> list[str]:
    seen, out = set(), []
    for name in names:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def _key_specs(result: ValidationResult) -> list[SpecRow]:
    rows = [SpecRow(name=k, value=v, source="input") for k, v in result.product.specs.items()]
    missing = _dedupe(result.empty_specs + result.missing_specs)
    rows += [SpecRow(name=k, value=NO_DATA, source="missing") for k in missing]
    return rows


WHO_FOR_MISSING = "Сферы применения (кому подходит)"


def _missing_fields(result: ValidationResult) -> list[str]:
    missing = []
    product = result.product
    if not product.manufacturer:
        missing.append("Производитель")
    if not product.description:
        missing.append("Описание")
    if not product.specs:
        missing.append("Технические характеристики")
    return _dedupe(missing + result.empty_specs + result.missing_specs)


def generate_card(raw: str | bytes | dict | ValidationResult, llm: JSONModel, max_repair_attempts: int = 1,
                  judge: Judge | None = None) -> ProductCard:
    """Главная функция. Бросает InputError (плохой вход) или LLMError (модель недоступна).

    judge — смысловая проверка; None — выключена. Сбой судьи карточку не ломает.
    """
    result = raw if isinstance(raw, ValidationResult) else parse_input(raw)
    product = result.product
    checker = FactChecker(_source_texts(product))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(product, result.missing_specs, result.sparse)},
    ]

    best: LLMCopy | None = None
    best_score = 0
    best_findings: list[Finding] = []
    problems: list[str] = []
    attempts = 0
    judge_active = judge is not None
    judge_error = ""
    for attempt in range(1, max_repair_attempts + 2):
        attempts = attempt
        raw_text = ""
        try:
            data, raw_text = llm.generate(messages)
            copy = _normalize(LLMCopy.model_validate(data), product)
            problems = review(copy, checker, product, result.sparse)
            findings: list[Finding] = []
            if judge_active:
                try:
                    findings = judge.review(product, copy)
                except LLMError as exc:
                    log.warning("Смысловая проверка не выполнена: %s", exc)
                    judge_active, judge_error = False, str(exc)
            problems = problems + [f.as_problem() for f in findings]
            # Запоминаем лучший корректный по схеме ответ (с наименьшим числом проблем).
            if best is None or len(problems) <= best_score:
                best, best_score, best_findings = copy, len(problems), findings
        except LLMResponseError as exc:
            raw_text, problems = exc.raw, [str(exc)]
        except ValidationError as exc:
            problems = ["ответ не соответствует схеме: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))} — {e['msg']}" for e in exc.errors()[:5])]

        if not problems:
            break
        log.info("Попытка %d: найдены проблемы: %s", attempt, problems)
        if attempt <= max_repair_attempts:
            messages = messages + [
                {"role": "assistant", "content": raw_text or "(пустой ответ)"},
                {"role": "user", "content": build_repair_prompt(problems)},
            ]

    if best is None:
        raise LLMError("Модель не вернула корректный ответ: " + "; ".join(problems))

    copy = best
    issues: list[Issue] = list(result.issues)
    if len(copy.seo_title) > TITLE_MAX:
        copy = copy.model_copy(update={"seo_title": _truncate_title(copy.seo_title)})
        issues.append(Issue(level="warning", field="seo_title",
                            message=f"Заголовок был длиннее {TITLE_MAX} символов и обрезан по слову — проверьте"))
    if result.sparse and len(copy.seo_description) < DESCRIPTION_MIN:
        issues.append(Issue(
            level="warning", field="seo_description",
            message=f"SEO-описание {len(copy.seo_description)} симв. — короче {DESCRIPTION_MIN}: во входных данных "
                    "мало сведений, текст намеренно не добивался выдумками. Дополните описание "
                    "(фрагмент паспорта) и сгенерируйте заново"))
    for problem in review(copy, checker, product, result.sparse):
        field = problem.split(":", 1)[0] if ":" in problem else "output"
        issues.append(Issue(level="warning", field=field, message="Проверьте: " + problem))
    final_texts = field_texts(copy)
    remaining = [f for f in best_findings if quote_found(f.quote, final_texts[f.field])]
    for finding in remaining:
        issues.append(Issue(level="warning", field=finding.field, message=finding.as_warning()))
    if judge is not None and judge_error:
        issues.append(Issue(level="warning", field="judge",
                            message="Смысловая проверка не выполнена: " + judge_error))
    for gap in copy.data_gaps:
        issues.append(Issue(level="info", field="data_gaps", message="Модель отмечает: " + gap))

    status = "needs_review" if any(i.level in {"error", "warning"} for i in issues) else "ok"
    return ProductCard(
        status=status,
        product={k: v for k, v in {
            "name": product.name,
            "manufacturer": product.manufacturer or NO_DATA,
            "product_type": product.product_type,
            "source_url": product.source_url,
            **{f"id:{k}": v for k, v in product.identifiers.items()},
        }.items() if v},
        seo_title=copy.seo_title,
        seo_description=copy.seo_description,
        key_specs=_key_specs(result),
        search_queries=copy.search_queries,
        commercial=copy.commercial,
        missing_fields=_missing_fields(result) + ([] if copy.commercial.who_for else [WHO_FOR_MISSING]),
        issues=issues,
        meta=Meta(
            model=getattr(llm, "model_name", "unknown"),
            generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            attempts=attempts,
            response_format=getattr(llm, "mode", "unknown"),
            prompt_version=PROMPT_VERSION,
            judge_model=judge.model_name if judge is not None else None,
            judge_status=("off" if judge is None else "error" if judge_error
                          else "issues" if remaining else "passed"),
        ),
    )
