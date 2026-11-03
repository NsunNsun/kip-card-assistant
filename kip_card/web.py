"""Веб-интерфейс и HTTP API.

    GET  /               — форма для контент-менеджера
    POST /api/generate   — JSON товара → карточка (JSON + Markdown)
    POST /api/check      — проверка входа без обращения к модели
    GET  /api/health     — состояние и текущая модель

Через POST /api/generate сервис можно вызывать из сайта на 1С-Битрикс или из 1С.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import ConfigError, Settings, load_settings
from .generator import generate_card
from .llm import LLMError, OpenAICompatibleLLM
from .render import to_markdown
from .validation import PROFILES, InputError, parse_input

STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).parent.parent / "examples" / "input"

app = FastAPI(title="kip-card-assistant", version=__version__,
              description="Черновики карточек товаров КИП с помощью LLM")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(get_settings())


def _error(status: int, message: str, issues=None) -> JSONResponse:
    body = {"error": message}
    if issues:
        body["issues"] = [i.model_dump() for i in issues]
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(InputError)
async def _input_error(_: Request, exc: InputError):
    return _error(422, "Некорректные входные данные", exc.issues)


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError):
    return _error(502, str(exc))


@app.exception_handler(ConfigError)
async def _config_error(_: Request, exc: ConfigError):
    return _error(503, str(exc))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    try:
        settings = get_settings()
    except ConfigError as exc:
        return {"status": "misconfigured", "error": str(exc), "version": __version__}
    return {"status": "ok", "model": settings.model, "llm_host": settings.host, "version": __version__}


@app.get("/api/examples")
def examples():
    """Примеры входных данных для формы."""
    if not EXAMPLES.is_dir():
        return []
    return [p.stem for p in sorted(EXAMPLES.glob("*.json"))]


@app.get("/api/examples/{name}")
def example(name: str):
    path = (EXAMPLES / f"{name}.json").resolve()
    if path.parent != EXAMPLES.resolve() or not path.is_file():
        return _error(404, "Пример не найден")
    return FileResponse(path, media_type="application/json")


async def _body(request: Request) -> bytes:
    return await request.body()


@app.post("/api/check")
def check(body: bytes = Depends(_body)):
    result = parse_input(body)
    return {
        "profile": result.profile,
        "profile_title": PROFILES[result.profile]["title"],
        "product": result.product.model_dump(),
        "missing_specs": result.missing_specs,
        "issues": [i.model_dump() for i in result.issues],
    }


def _llm_dependency() -> OpenAICompatibleLLM:
    return get_llm()


@app.post("/api/generate")
def generate(body: bytes = Depends(_body), llm=Depends(_llm_dependency)):
    result = parse_input(body)  # сначала вход: плохой JSON не должен тратить токены
    settings = getattr(llm, "settings", None)
    attempts = settings.max_repair_attempts if settings else 1
    card = generate_card(result, llm, attempts)
    return {"card": card.model_dump(), "markdown": to_markdown(card)}
