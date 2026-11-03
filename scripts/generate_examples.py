"""Пакетная генерация карточек из JSON-файлов.

Примеры:

    python scripts/generate_examples.py
    python scripts/generate_examples.py --input data/products/pr200.json --out examples
    python scripts/generate_examples.py --format json

Скрипт печатает статус проверки достоверности и пишет результат в ``examples/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, SettingsError  # noqa: E402
from app.generator import GenerationResult, generate_card  # noqa: E402
from app.llm import LLMClient, LLMError  # noqa: E402
from app.render import to_markdown  # noqa: E402
from app.validation import InputError, parse_product  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data" / "products"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация карточек товаров через LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON-файл товара, директория с JSON или glob-шаблон",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="куда писать результат")
    parser.add_argument(
        "--format",
        choices=("both", "json", "markdown"),
        default="both",
        help="формат сохраняемых файлов",
    )
    return parser.parse_args(argv)


def collect_inputs(target: Path) -> list[Path]:
    """Разворачивает путь в список JSON-файлов товаров."""
    if target.is_dir():
        return sorted(path for path in target.glob("*.json") if path.is_file())
    if target.is_file():
        return [target]
    matches = sorted(Path().glob(str(target)))
    if not matches:
        raise SystemExit(f"Не найдено ни одного входного файла: {target}")
    return matches


async def generate_one(
    path: Path, settings: Settings, client: LLMClient, output: Path, fmt: str
) -> tuple[str, GenerationResult | None, str | None]:
    """Генерирует карточку для одного файла и сохраняет результат."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return path.stem, None, f"невалидный JSON: {exc}"

    try:
        analysis = parse_product(raw)
    except InputError as exc:
        return path.stem, None, str(exc)

    try:
        result = await generate_card(analysis, client)
    except LLMError as exc:
        return path.stem, None, f"{exc.code}: {exc}"

    output.mkdir(parents=True, exist_ok=True)
    stem = analysis.product.id or path.stem
    if fmt in {"both", "json"}:
        (output / f"{stem}.json").write_text(
            json.dumps(result.card.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if fmt in {"both", "markdown"}:
        (output / f"{stem}.md").write_text(to_markdown(result.card), encoding="utf-8")
    return stem, result, None


async def run(args: argparse.Namespace) -> int:
    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        print(f"Ошибка настроек: {exc}", file=sys.stderr)
        return 2

    if not settings.is_configured:
        print(
            "LLM_API_KEY не задан. Скопируйте .env.example в .env и укажите ключ API.",
            file=sys.stderr,
        )
        return 2

    paths = collect_inputs(args.input)
    if not paths:
        print(f"В {args.input} нет JSON-файлов", file=sys.stderr)
        return 2

    client = LLMClient(settings)
    failures = 0

    print(f"Модель: {settings.model} ({settings.base_url})")
    print(f"Товаров: {len(paths)}\n")

    for path in paths:
        stem, result, error = await generate_one(
            path, settings, client, args.out, args.format
        )
        if error or result is None:
            failures += 1
            print(f"✗ {stem}: {error}")
            continue

        card = result.card
        fact_check = card.fact_check
        mark = "✓" if fact_check.get("passed") else "⚠"
        print(
            f"{mark} {stem}: заголовок {len(card.seo_title)}/70, "
            f"описание {len(card.seo_description)} симв., "
            f"характеристик {len(card.specs)}, запросов {len(card.search_queries)}, "
            f"проверка {'пройдена' if fact_check.get('passed') else 'с замечаниями'}"
        )
        for warning in card.warnings:
            print(f"    · {warning}")
        print(f"    → {args.out}")

    print()
    if failures:
        print(f"Ошибок: {failures} из {len(paths)}", file=sys.stderr)
        return 1
    print(f"Готово: {len(paths)} карточки в {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
