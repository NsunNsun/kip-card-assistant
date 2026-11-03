"""Вывод карточки в Markdown — удобно читать и копировать контент-менеджеру."""

from __future__ import annotations

from .models import ProductCard

_LEVEL = {"error": "Ошибка", "warning": "Проверить", "info": "Заметка"}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(card: ProductCard) -> str:
    p = card.product
    lines: list[str] = []
    lines.append(f"# {p.get('name', '')}")
    status = "готово к вычитке" if card.status == "ok" else "требует проверки"
    lines.append(
        f"*Черновик карточки · {status} · модель `{card.meta.model}` · {card.meta.generated_at}*"
    )
    if p.get("source_url"):
        lines.append(f"\nИсточник данных: {p['source_url']}")

    attention = [i for i in card.issues if i.level != "info"]
    if attention or card.missing_fields:
        lines.append("\n## Требует внимания")
        if card.missing_fields:
            lines.append("**Нет данных:** " + ", ".join(card.missing_fields))
        for issue in attention:
            lines.append(f"- {_LEVEL[issue.level]}: {issue.message}")

    lines.append(f"\n## SEO-заголовок ({len(card.seo_title)}/70)")
    lines.append(card.seo_title)

    lines.append(f"\n## SEO-описание ({len(card.seo_description)} симв.)")
    lines.append(card.seo_description)

    lines.append("\n## Ключевые характеристики")
    lines.append("| Характеристика | Значение |")
    lines.append("|---|---|")
    for row in card.key_specs:
        value = f"**{row.value}**" if row.source == "missing" else row.value
        lines.append(f"| {_cell(row.name)} | {_cell(value)} |")

    lines.append("\n## Поисковые запросы")
    for i, query in enumerate(card.search_queries, 1):
        lines.append(f"{i}. {query}")

    lines.append("\n## Для менеджера")
    lines.append(card.commercial.summary)
    lines.append("\n**Кому подходит:**")
    lines += [f"- {x}" for x in card.commercial.who_for]
    lines.append("\n**Какие задачи решает:**")
    lines += [f"- {x}" for x in card.commercial.tasks]

    notes = [i for i in card.issues if i.level == "info"]
    if notes:
        lines.append("\n## Заметки")
        lines += [f"- {i.message}" for i in notes]

    return "\n".join(lines) + "\n"
