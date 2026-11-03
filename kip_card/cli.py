"""Командная строка.

    python -m kip_card generate examples/input/testo-830-t1.json
    python -m kip_card generate examples/input/*.json -o examples/output
    python -m kip_card check examples/input/manometer-dm02.json      # без LLM
    python -m kip_card import-yml feed.xml --offer-id 64423 -o examples/input
    python -m kip_card audit-yml feed.xml
    python -m kip_card serve
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_settings
from .generator import generate_card
from .llm import LLMError, OpenAICompatibleLLM
from .render import to_markdown
from .validation import PROFILES, InputError, parse_input
from .yml_import import FeedError, audit, find_offers, load_feed

EXIT_OK, EXIT_INPUT, EXIT_LLM = 0, 1, 2


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8-sig")


def expand_paths(patterns: list[str]) -> tuple[list[str], list[str]]:
    """Раскрывает шаблоны вида examples/input/*.json.

    bash раскрывает их сам, а PowerShell и cmd.exe передают звёздочку программе как есть.
    Возвращает (файлы, шаблоны без совпадений).
    """
    files: list[str] = []
    unmatched: list[str] = []
    for pattern in patterns:
        if pattern == "-" or Path(pattern).exists() or not any(ch in pattern for ch in "*?["):
            files.append(pattern)
            continue
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        else:
            unmatched.append(pattern)
    return files, unmatched


def _report_unmatched(unmatched: list[str]) -> int:
    for pattern in unmatched:
        print(f"→ {pattern}\n  Ошибка входных данных: по шаблону не найдено ни одного файла", file=sys.stderr)
    return EXIT_INPUT if unmatched else EXIT_OK


def _print_issues(prefix: str, issues) -> None:
    for issue in issues:
        print(f"{prefix}[{issue.level}] {issue.field}: {issue.message}", file=sys.stderr)


def cmd_generate(args) -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Ошибка настройки: {exc}", file=sys.stderr)
        return EXIT_LLM
    llm = OpenAICompatibleLLM(settings)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    files, unmatched = expand_paths(args.files)
    code = _report_unmatched(unmatched)
    for path in files:
        label = "stdin" if path == "-" else path
        print(f"→ {label}", file=sys.stderr)
        try:
            card = generate_card(_read(path), llm, settings.max_repair_attempts)
        except (OSError, InputError) as exc:
            print(f"  Ошибка входных данных: {exc}", file=sys.stderr)
            code = max(code, EXIT_INPUT)
            continue
        except LLMError as exc:
            print(f"  Ошибка модели: {exc}", file=sys.stderr)
            return EXIT_LLM

        markdown = to_markdown(card)
        as_json = card.model_dump_json(indent=2)
        if out_dir:
            stem = "stdin" if path == "-" else Path(path).stem
            if args.format in ("md", "both"):
                (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
            if args.format in ("json", "both"):
                (out_dir / f"{stem}.json").write_text(as_json + "\n", encoding="utf-8")
            print(f"  {card.status} · попыток: {card.meta.attempts} · сохранено в {out_dir / stem}.*",
                  file=sys.stderr)
        else:
            print(as_json if args.format == "json" else markdown)
    return code


def cmd_check(args) -> int:
    files, unmatched = expand_paths(args.files)
    code = _report_unmatched(unmatched)
    for path in files:
        print(f"→ {path}", file=sys.stderr)
        try:
            result = parse_input(_read(path))
        except (OSError, InputError) as exc:
            issues = getattr(exc, "issues", None)
            if issues:
                _print_issues("  ", issues)
            else:
                print(f"  Ошибка: {exc}", file=sys.stderr)
            code = EXIT_INPUT
            continue
        print(f"  Тип прибора: {PROFILES[result.profile]['title']}; характеристик: {len(result.product.specs)}",
              file=sys.stderr)
        _print_issues("  ", result.issues)
        if not result.issues:
            print("  Замечаний нет", file=sys.stderr)
    return code


def cmd_import_yml(args) -> int:
    try:
        products = find_offers(load_feed(args.feed), args.offer_id)
    except FeedError as exc:
        print(f"Ошибка фида: {exc}", file=sys.stderr)
        return EXIT_INPUT
    for product in products:
        text = json.dumps(product, ensure_ascii=False, indent=2)
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"offer-{product['identifiers']['id в YML-фиде']}.json"
            target.write_text(text + "\n", encoding="utf-8")
            print(f"Сохранено: {target}", file=sys.stderr)
        else:
            print(text)
    return EXIT_OK


def cmd_audit_yml(args) -> int:
    try:
        stats = audit(load_feed(args.feed))
    except FeedError as exc:
        print(f"Ошибка фида: {exc}", file=sys.stderr)
        return EXIT_INPUT
    total = stats["offers"] or 1
    print(f"Товаров в фиде: {stats['offers']}")
    print(f"С описанием: {stats['with_description']} ({stats['with_description'] / total:.0%})")
    print(f"С техническими характеристиками: {stats['with_specs']} ({stats['with_specs'] / total:.0%})")
    rows = stats["without_both_by_category"][: args.top]
    if rows:
        print(f"\nБез описания и характеристик, топ-{len(rows)} категорий:")
        for name, count in rows:
            print(f"  {count:>6}  {name}")
    return EXIT_OK


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("kip_card.web:app", host=args.host, port=args.port)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kip_card",
                                     description="Черновики карточек товаров КИП с помощью LLM")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="сгенерировать карточку по JSON")
    p.add_argument("files", nargs="+", help="JSON-файлы с данными о товаре ('-' — stdin)")
    p.add_argument("-o", "--out", help="папка для результатов (иначе вывод в консоль)")
    p.add_argument("-f", "--format", choices=["md", "json", "both"], default="both",
                   help="формат результата (по умолчанию в файлы пишутся оба; в консоль — Markdown)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("check", help="проверить входной JSON без обращения к модели")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("import-yml", help="создать входной JSON из товара YML-фида")
    p.add_argument("feed", help="путь к фиду или URL")
    p.add_argument("--offer-id", action="append", required=True, help="id товара (можно несколько раз)")
    p.add_argument("-o", "--out", help="папка для JSON")
    p.set_defaults(func=cmd_import_yml)

    p = sub.add_parser("audit-yml", help="сводка полноты данных в YML-фиде")
    p.add_argument("feed", help="путь к фиду или URL")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_audit_yml)

    p = sub.add_parser("serve", help="запустить веб-форму")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return args.func(args)
