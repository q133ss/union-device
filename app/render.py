"""Представление карточки: JSON для интеграций и Markdown для копирования."""

from __future__ import annotations

import json
from typing import Iterable

from app.models import ProductCard
from app.prompts import DESCRIPTION_MAX, DESCRIPTION_MIN, MAX_TITLE_LENGTH

NO_DATA = "данных нет"


def to_json(card: ProductCard, *, indent: int = 2) -> str:
    """Сериализует карточку в JSON."""
    return json.dumps(card.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def to_markdown(card: ProductCard) -> str:
    """Собирает Markdown-версию карточки, готовую для копирования."""
    manufacturer = card.manufacturer or NO_DATA
    lines: list[str] = []

    lines.append(f"# {card.seo_title or card.name}")
    lines.append("")
    lines.append(f"**Производитель:** {manufacturer}  ")
    lines.append(f"**Модель:** {card.name}  ")
    if card.source_url:
        lines.append(f"**Источник данных:** {card.source_url}  ")
    if card.id:
        lines.append(f"**Код:** {card.id}  ")
    lines.append("")

    lines.append(f"## SEO-заголовок ({len(card.seo_title)}/{MAX_TITLE_LENGTH})")
    lines.append("")
    lines.append(card.seo_title or NO_DATA)
    lines.append("")

    description = card.seo_description or NO_DATA
    lines.append(
        f"## SEO-описание ({len(card.seo_description)} символов, "
        f"норма {DESCRIPTION_MIN}–{DESCRIPTION_MAX})"
    )
    lines.append("")
    lines.append(description)
    lines.append("")

    lines.append("## Характеристики")
    lines.append("")
    if card.specs:
        lines.append("| Характеристика | Значение |")
        lines.append("| --- | --- |")
        for spec in card.specs:
            lines.append(f"| {_cell(spec.name)} | {_cell(spec.value)} |")
    else:
        lines.append(f"Характеристики: {NO_DATA}.")
    lines.append("")

    lines.append("## Поисковые запросы")
    lines.append("")
    if card.search_queries:
        for index, query in enumerate(card.search_queries, start=1):
            lines.append(f"{index}. {query}")
    else:
        lines.append(f"Поисковые запросы: {NO_DATA}.")
    lines.append("")

    lines.append("## Commercial-блок для менеджера")
    lines.append("")
    lines.append(card.commercial_block or NO_DATA)
    lines.append("")

    if card.data_gaps:
        lines.append("## Чего не хватает в данных")
        lines.append("")
        lines.extend(f"- {gap} — {NO_DATA}" for gap in card.data_gaps)
        lines.append("")

    if card.warnings:
        lines.append("## Предупреждения")
        lines.append("")
        lines.extend(f"- {warning}" for warning in card.warnings)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.extend(_meta_lines(card))
    lines.append("")
    return "\n".join(lines)


def to_text(card: ProductCard) -> str:
    """Плоский текст для вставки в письмо или мессенджер."""
    parts = [
        card.seo_title or card.name,
        "",
        card.seo_description or NO_DATA,
        "",
        "Характеристики:",
    ]
    parts.extend(f"- {spec.name}: {spec.value}" for spec in card.specs)
    if card.search_queries:
        parts.append("")
        parts.append("Поисковые запросы: " + "; ".join(card.search_queries))
    if card.commercial_block:
        parts.append("")
        parts.append(card.commercial_block)
    return "\n".join(parts)


def _cell(value: str) -> str:
    """Экранирует вертикальную черту, чтобы не ломать таблицу Markdown."""
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _meta_lines(card: ProductCard) -> Iterable[str]:
    """Служебная строка: статус проверки, модель и расход токенов."""
    fact_check = card.fact_check or {}
    passed = fact_check.get("passed")
    status = "✅ проверка достоверности пройдена" if passed else "⚠️ есть замечания проверки достоверности"
    yield status

    llm = (card.meta or {}).get("llm") or {}
    elapsed = (card.meta or {}).get("elapsed_seconds")
    if elapsed is None:
        elapsed = (llm.get("latency_ms") or 0) / 1000
    if llm:
        yield (
            f"Модель: {llm.get('model') or '—'} · вызовов: {llm.get('calls', 1)} · "
            f"токенов: {llm.get('total_tokens', 0)} · {float(elapsed):.1f} c"
        )
    generated_at = (card.meta or {}).get("generated_at")
    if generated_at:
        yield f"Сгенерировано: {generated_at}"
