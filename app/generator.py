"""Оркестрация генерации: данные → промпт → LLM → проверка → карточка."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.factcheck import FactCheckReport, check, sanitize
from app.llm import LLMClient, LLMError, LLMResult
from app.models import CardDraft, ProductCard, SpecItem
from app.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from app.validation import Analysis

MAX_REPAIR_ATTEMPTS = 1


@dataclass(frozen=True)
class GenerationResult:
    """Результат генерации: карточка, отчёт проверки и статистика."""

    card: ProductCard
    report: FactCheckReport
    repair_attempts: int


async def generate_card(analysis: Analysis, client: LLMClient) -> GenerationResult:
    """Строит карточку товара и проверяет её на выдуманные данные.

    Схема: запрос → проверка → (при нарушениях) один запрос на исправление →
    вычищение неподтверждённых строк → итоговая карточка с отчётом.
    """
    product = analysis.product
    raw_draft, result = await client.complete_json(
        SYSTEM_PROMPT, build_user_prompt(product, analysis.gaps)
    )
    draft = _to_draft(raw_draft)
    report = check(draft, product, analysis.gaps)

    usage = _Usage.from_result(result)
    repair_attempts = 0

    while report.requires_repair and repair_attempts < MAX_REPAIR_ATTEMPTS:
        repair_attempts += 1
        violations = [violation.message for violation in report.errors]
        raw_draft, repair_result = await client.complete_json(
            SYSTEM_PROMPT, build_repair_prompt(product, draft.model_dump(), violations)
        )
        draft = _to_draft(raw_draft)
        usage = usage.plus(repair_result)
        report = check(draft, product, analysis.gaps)

    final_draft = sanitize(draft, report, analysis.gaps)
    final_report = check(final_draft, product, analysis.gaps)
    card = _build_card(product, final_draft, final_report, usage, repair_attempts, analysis)
    return GenerationResult(card=card, report=final_report, repair_attempts=repair_attempts)


# ---------------------------------------------------------------------- helpers


def _to_draft(raw: dict[str, Any]) -> CardDraft:
    """Приводит ответ модели к терпимой модели карточки.

    Если модель вернула не объект или поля неверного типа, это ошибка ответа
    LLM (502), а не падение сервиса.
    """
    try:
        return CardDraft.model_validate(raw)
    except ValidationError as exc:
        raise LLMError(
            "Модель вернула JSON не по ожидаемой схеме карточки",
            code="llm_invalid_json",
            details={"errors": [error.get("msg", "") for error in exc.errors()][:5]},
        ) from exc


@dataclass(frozen=True)
class _Usage:
    """Накопленная статистика по всем вызовам модели."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    calls: int = 0

    @classmethod
    def from_result(cls, result: LLMResult) -> "_Usage":
        return cls(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            calls=1,
        )

    def plus(self, result: LLMResult) -> "_Usage":
        return _Usage(
            model=result.model or self.model,
            prompt_tokens=self.prompt_tokens + result.prompt_tokens,
            completion_tokens=self.completion_tokens + result.completion_tokens,
            latency_ms=self.latency_ms + result.latency_ms,
            calls=self.calls + 1,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_ms": self.latency_ms,
            "calls": self.calls,
        }


def _build_card(
    product,
    draft: CardDraft,
    report: FactCheckReport,
    usage: _Usage,
    repair_attempts: int,
    analysis: Analysis,
) -> ProductCard:
    warnings = [warning.strip() for warning in draft.warnings if warning.strip()]
    for error in report.errors:
        warnings.append(f"Требует проверки: {error.message}")

    meta = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "llm": usage.as_dict(),
        "repair_attempts": repair_attempts,
        "gaps_detected": [gap.label for gap in analysis.gaps],
        "input_notes": list(analysis.notes),
    }

    return ProductCard(
        id=product.id,
        name=product.name,
        manufacturer=product.manufacturer or "",
        source_url=product.source_url,
        seo_title=draft.seo_title.strip(),
        seo_description=draft.seo_description.strip(),
        specs=[SpecItem(name=item.name, value=item.value) for item in draft.specs if item.name],
        search_queries=[query for query in draft.search_queries if query.strip()],
        commercial_block=draft.commercial_block.strip(),
        data_gaps=[gap for gap in draft.data_gaps if gap.strip()],
        warnings=warnings,
        fact_check=report.as_dict(),
        meta=meta,
    )
