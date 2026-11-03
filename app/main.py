"""HTTP-сервис: веб-форма для контент-менеджера и JSON API.

Запуск: ``uvicorn app.main:app --reload``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import Settings
from app.generator import GenerationResult, generate_card
from app.llm import LLMClient, LLMError
from app.render import to_markdown
from app.validation import Analysis, FieldIssue, InputError, parse_product

logger = logging.getLogger("union_device")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR.parent / "data" / "products"

MAX_BATCH_SIZE = 10
FORMATS = {"json", "markdown", "both"}

# Коды ошибок LLM → HTTP-статусы для клиента API.
LLM_STATUS = {
    "llm_not_configured": 503,
    "llm_auth": 502,
    "llm_rate_limit": 429,
    "llm_timeout": 504,
    "llm_unavailable": 502,
    "llm_bad_request": 502,
    "llm_invalid_json": 502,
}


def create_app(
    settings: Settings | None = None,
    client_factory: Callable[[Settings], LLMClient] | None = None,
) -> FastAPI:
    """Создаёт приложение. Зависимости можно подменить в тестах."""
    app = FastAPI(
        title="union-device",
        version=__version__,
        description="AI-помощник для подготовки карточек КИП для B2B-магазина",
    )
    resolved_settings = settings or Settings.from_env()
    app.state.settings = resolved_settings
    app.state.client_factory = client_factory or (lambda cfg: LLMClient(cfg))
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # ------------------------------------------------------------------ ошибки

    @app.exception_handler(InputError)
    async def _handle_input_error(_: Request, exc: InputError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_input",
                    "message": "Некорректные входные данные",
                    "details": {"issues": [issue.as_dict() for issue in exc.issues]},
                }
            },
        )

    @app.exception_handler(LLMError)
    async def _handle_llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(
            status_code=LLM_STATUS.get(exc.code, 502), content={"error": exc.as_dict()}
        )

    # ----------------------------------------------------------------- страницы

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "example_json": _example_json(),
                "model": resolved_settings.model,
                "base_url": resolved_settings.base_url,
                "llm_configured": resolved_settings.is_configured,
                "version": __version__,
                "products": _available_products(),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "llm_configured": resolved_settings.is_configured,
            "model": resolved_settings.model,
            "base_url": resolved_settings.base_url,
        }

    @app.get("/api/example")
    async def example(product: str | None = None) -> Response:
        return Response(_example_json(product), media_type="application/json; charset=utf-8")

    # --------------------------------------------------------------------- API

    @app.post("/api/cards")
    async def create_card(request: Request) -> Response:
        raw = await _read_json(request)
        fmt = request.query_params.get("format", "both")
        if fmt not in FORMATS:
            raise InputError(
                [FieldIssue("format", f"допустимые значения: {', '.join(sorted(FORMATS))}")]
            )

        analysis = parse_product(raw)
        result = await _generate(analysis, resolved_settings, app.state.client_factory)
        logger.info(
            "card generated: product=%s model=%s passed=%s",
            analysis.product.name,
            result.card.meta.get("llm", {}).get("model"),
            result.card.fact_check.get("passed"),
        )
        return _render_result(result, fmt)

    @app.post("/api/cards/batch")
    async def create_cards(request: Request) -> JSONResponse:
        raw = await _read_json(request)
        if isinstance(raw, dict) and "products" in raw:
            items = raw["products"]
        else:
            items = raw
        if not isinstance(items, list):
            raise InputError(
                [FieldIssue("products", "ожидался массив товаров или объект с полем products")]
            )
        if not items:
            raise InputError([FieldIssue("products", "массив товаров пуст")])
        if len(items) > MAX_BATCH_SIZE:
            raise InputError(
                [
                    FieldIssue(
                        "products",
                        f"за один запрос обрабатывается не больше {MAX_BATCH_SIZE} товаров",
                    )
                ]
            )

        results: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            try:
                analysis = parse_product(item)
            except InputError as exc:
                results.append(
                    {
                        "index": index,
                        "error": {
                            "code": "invalid_input",
                            "message": "Некорректные входные данные",
                            "details": {"issues": [issue.as_dict() for issue in exc.issues]},
                        },
                    }
                )
                continue
            try:
                result = await _generate(analysis, resolved_settings, app.state.client_factory)
            except LLMError as exc:
                results.append({"index": index, "error": exc.as_dict()})
                continue
            results.append(
                {
                    "index": index,
                    "card": result.card.model_dump(mode="json"),
                    "markdown": to_markdown(result.card),
                }
            )

        return JSONResponse({"results": results})

    return app


async def _generate(
    analysis: Analysis,
    settings: Settings,
    client_factory: Callable[[Settings], LLMClient],
) -> GenerationResult:
    client = client_factory(settings)
    return await generate_card(analysis, client)


async def _read_json(request: Request) -> Any:
    """Читает тело запроса и превращает ошибки парсинга в понятные 422."""
    body = await request.body()
    if not body.strip():
        raise InputError([FieldIssue("body", "пустое тело запроса: ожидался JSON-объект товара")])
    try:
        return json.loads(body)
    except ValueError as exc:
        raise InputError(
            [FieldIssue("body", f"тело запроса не является валидным JSON: {exc}")]
        ) from exc


def _render_result(result: GenerationResult, fmt: str) -> Response:
    card = result.card
    if fmt == "markdown":
        return PlainTextResponse(to_markdown(card), media_type="text/markdown; charset=utf-8")
    if fmt == "json":
        return JSONResponse(card.model_dump(mode="json"))
    return JSONResponse(
        {
            "card": card.model_dump(mode="json"),
            "markdown": to_markdown(card),
            "report": card.fact_check,
        }
    )


def _example_json(product: str | None = None) -> str:
    """Пример входных данных для формы — реальный товар из data/products."""
    path = _product_path(product) if product else DATA_DIR / "pr200.json"
    if path.is_file():
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    return json.dumps(
        {"name": "Манометр показывающий DM 02", "manufacturer": "Манотомь", "specs": {}},
        ensure_ascii=False,
        indent=2,
    )


def _product_path(product_id: str) -> Path:
    """Безопасно превращает идентификатор товара в путь внутри data/products."""
    safe = "".join(char for char in product_id if char.isalnum() or char in "-_")
    return DATA_DIR / f"{safe}.json"


def _available_products() -> list[dict[str, str]]:
    """Список товаров из data/products — для кнопок быстрой загрузки в форме."""
    products: list[dict[str, str]] = []
    for path in sorted(DATA_DIR.glob("*.json")) if DATA_DIR.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        products.append({"id": path.stem, "name": str(data.get("name") or path.stem)})
    return products


app = create_app()
