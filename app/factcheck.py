"""Детерминированная проверка карточки на выдуманные данные.

Модель может «дописать» характеристику или цифру, которых нет во входных данных.
Полагаться на аккуратность промпта нельзя, поэтому каждое значение проверяется
кодом:

* каждая характеристика обязана дословно (без учёта регистра, пробелов и
  пунктуации) встречаться во входных данных;
* каждое число в заголовке, описании, commercial-блоке и запросах должно быть
  во входных данных — иначе это выдумка;
* характеристики, которые по чек-листу отсутствуют, не могут получить значение;
* объёмы (70 / 800–1200 / 3–5 / 300–600 символов) проверяются по факту.

Всё, что не прошло проверку, либо уходит в повторный запрос к модели, либо
удаляется из карточки — но никогда не остаётся в результате как факт.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.models import CardDraft, ProductInput
from app.normalize import lower, numbers, squeeze
from app.prompts import (
    COMMERCIAL_MAX,
    COMMERCIAL_MIN,
    DESCRIPTION_MAX,
    DESCRIPTION_MIN,
    MAX_TITLE_LENGTH,
    QUERIES_MAX,
    QUERIES_MIN,
)
from app.validation import SPEC_GROUPS, Gap

GAP_MARKER = "данных нет"
MIN_SPECS = 6
MAX_SPECS = 12
MAX_QUERY_LENGTH = 120

_WORD = re.compile(r"[а-яa-z]{3,}")


@dataclass(frozen=True)
class Violation:
    """Одно нарушение правил карточки."""

    code: str
    message: str
    severity: str = "error"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True)
class FactCheckReport:
    """Итог проверки: что именно не так и какие строки характеристик удалить."""

    violations: tuple[Violation, ...] = ()
    invalid_specs: tuple[int, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Violation]:
        return [item for item in self.violations if item.is_error]

    @property
    def warnings(self) -> list[Violation]:
        return [item for item in self.violations if not item.is_error]

    @property
    def requires_repair(self) -> bool:
        return bool(self.errors)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": [{"code": item.code, "message": item.message} for item in self.errors],
            "warnings": [{"code": item.code, "message": item.message} for item in self.warnings],
            "stats": self.stats,
        }


@dataclass(frozen=True)
class SourceFacts:
    """Сжатое представление входных данных для быстрых сравнений."""

    squeezed: str
    words: frozenset[str]
    stems: frozenset[str]
    numbers: frozenset[str]

    @classmethod
    def from_product(cls, product: ProductInput) -> "SourceFacts":
        text = product.flat_text
        words = _WORD.findall(lower(text))
        return cls(
            squeezed=squeeze(text),
            words=frozenset(words),
            stems=frozenset(word[:5] for word in words),
            numbers=frozenset(numbers(text)),
        )

    def supports(self, name: str, value: str) -> bool:
        """Проверяет, что пара «характеристика → значение» подтверждена входом.

        Проверяются обе части: и название характеристики, и её значение.
        Одного значения мало — иначе «Класс точности: 1,5» пройдёт проверку
        только потому, что «1,5» встречается в обозначении резьбы М20×1,5.
        """
        if GAP_MARKER in lower(value):
            return True
        return self._value_supported(value) and self._name_supported(name)

    def _value_supported(self, value: str) -> bool:
        """Значение целиком собрано из входных данных."""
        candidate = squeeze(value)
        if not candidate:
            return False
        if len(candidate) >= 4 and candidate in self.squeezed:
            return True

        value_numbers = numbers(value)
        if not value_numbers.issubset(self.numbers):
            return False

        value_words = _WORD.findall(lower(value))
        if not value_words:
            # Значение без слов (число, обозначение) подтверждено номерами.
            return bool(value_numbers)

        return all(word in self.words for word in value_words)

    def _name_supported(self, name: str) -> bool:
        """Название характеристики встречается во входных данных хотя бы наполовину.

        Сравнение по основам слов (первые 5 букв), чтобы «Интерфейсы связи» и
        «Интерфейс RS-485» считались одной характеристикой.
        """
        name_words = _WORD.findall(lower(name))
        if not name_words:
            return True
        name_stems = {word[:5] for word in name_words}
        hits = sum(1 for stem in name_stems if stem in self.stems)
        return hits * 2 >= len(name_stems)


def check(draft: CardDraft, product: ProductInput, gaps: Sequence[Gap] = ()) -> FactCheckReport:
    """Проверяет черновик целиком и возвращает отчёт."""
    source = SourceFacts.from_product(product)
    violations: list[Violation] = []
    invalid_specs: list[int] = []

    invalid_specs.extend(_check_specs(draft, source, violations))
    _check_gaps(draft, gaps, violations)
    _check_numbers(draft, source, violations)
    _check_volumes(draft, violations)
    _check_language(draft, violations)

    stats = {
        "specs_checked": len(draft.specs),
        "specs_rejected": len(invalid_specs),
        "description_chars": len(draft.seo_description),
        "title_chars": len(draft.seo_title),
        "queries": len(draft.search_queries),
    }
    return FactCheckReport(
        violations=tuple(violations),
        invalid_specs=tuple(sorted(set(invalid_specs))),
        stats=stats,
    )


def sanitize(draft: CardDraft, report: FactCheckReport, gaps: Sequence[Gap] = ()) -> CardDraft:
    """Убирает из карточки всё, что не подтверждено, и подтягивает объёмы.

    Описание не дописывается: если модель не добрала символов, честнее оставить
    предупреждение, чем сочинять текст без источника.
    """
    updates: dict[str, Any] = {}
    warnings = [item.strip() for item in draft.warnings if item.strip()]

    rejected = set(report.invalid_specs)
    if rejected:
        kept = [spec for index, spec in enumerate(draft.specs) if index not in rejected]
        dropped = [spec.name or spec.value for index, spec in enumerate(draft.specs) if index in rejected]
        updates["specs"] = kept
        warnings.append(
            "Удалены характеристики без подтверждения во входных данных: " + ", ".join(dropped)
        )

    queries = _dedupe_queries(draft.search_queries)[:QUERIES_MAX]
    if queries != list(draft.search_queries):
        updates["search_queries"] = queries

    if len(draft.seo_title) > MAX_TITLE_LENGTH:
        updates["seo_title"] = _trim_title(draft.seo_title)
        warnings.append(f"Заголовок сокращён до {MAX_TITLE_LENGTH} символов")

    if len(draft.seo_description) > DESCRIPTION_MAX:
        updates["seo_description"] = _trim_to(draft.seo_description, DESCRIPTION_MAX, DESCRIPTION_MIN)
        warnings.append(f"Описание сокращено до {DESCRIPTION_MAX} символов")

    if len(draft.commercial_block) > COMMERCIAL_MAX:
        updates["commercial_block"] = _trim_to(
            draft.commercial_block, COMMERCIAL_MAX, COMMERCIAL_MIN
        )
        warnings.append(f"Commercial-блок сокращён до {COMMERCIAL_MAX} символов")

    merged_gaps = _merge_gaps(draft.data_gaps, gaps)
    if merged_gaps != list(draft.data_gaps):
        updates["data_gaps"] = merged_gaps

    updates["warnings"] = warnings
    return draft.model_copy(update=updates)


# ------------------------------------------------------------------ отдельные проверки


def _check_specs(draft: CardDraft, source: SourceFacts, violations: list[Violation]) -> list[int]:
    """Характеристики должны быть подтверждены входными данными."""
    invalid: list[int] = []
    for index, spec in enumerate(draft.specs):
        if not spec.name or not spec.value:
            continue
        if source.supports(spec.name, spec.value):
            continue
        invalid.append(index)
        violations.append(
            Violation(
                "spec_not_in_source",
                f"Значение характеристики «{spec.name}» = «{spec.value}» отсутствует "
                "во входных данных — такую строку нужно убрать или заменить на «данных нет»",
            )
        )
    return invalid


def _check_gaps(draft: CardDraft, gaps: Sequence[Gap], violations: list[Violation]) -> None:
    """Отсутствующие по чек-листу характеристики не должны получить значение."""
    if not gaps:
        return
    groups = {group.key: group for group in SPEC_GROUPS}
    missing = [groups[gap.key] for gap in gaps if gap.key in groups]
    for spec in draft.specs:
        name = lower(spec.name)
        if GAP_MARKER in lower(spec.value):
            continue
        for group in missing:
            if any(re.search(pattern, name) for pattern in group.patterns):
                violations.append(
                    Violation(
                        "gap_filled",
                        f"Характеристика «{group.label}» отсутствует во входных данных, "
                        f"но получила значение «{spec.value}» — нужно «{GAP_MARKER}»",
                    )
                )
                break


def _check_numbers(draft: CardDraft, source: SourceFacts, violations: list[Violation]) -> None:
    """Любое число в текстах карточки должно встречаться во входных данных."""
    texts: list[tuple[str, str]] = [
        ("заголовке", draft.seo_title),
        ("описании", draft.seo_description),
        ("commercial-блоке", draft.commercial_block),
    ]
    texts.extend(("поисковом запросе", query) for query in draft.search_queries)

    reported: set[str] = set()
    for place, text in texts:
        for token in sorted(numbers(text) - source.numbers):
            if token in reported:
                continue
            reported.add(token)
            violations.append(
                Violation(
                    "number_not_in_source",
                    f"Число «{token}» в {place} не встречается во входных данных — "
                    "уберите его или замените на формулировку без цифр",
                )
            )


def _check_volumes(draft: CardDraft, violations: list[Violation]) -> None:
    """Объёмы и количество элементов — по требованиям задания."""
    title = draft.seo_title.strip()
    if not title:
        violations.append(Violation("title_missing", "Не заполнен SEO-заголовок"))
    elif len(title) > MAX_TITLE_LENGTH:
        violations.append(
            Violation(
                "title_too_long",
                f"Заголовок длиннее {MAX_TITLE_LENGTH} символов ({len(title)}) — сократите",
            )
        )

    description = draft.seo_description.strip()
    if not description:
        violations.append(Violation("description_missing", "Не заполнено SEO-описание"))
    elif len(description) < DESCRIPTION_MIN:
        violations.append(
            Violation(
                "description_too_short",
                f"SEO-описание короче {DESCRIPTION_MIN} символов ({len(description)}) — "
                "расширьте за счёт назначения и областей применения из данных",
            )
        )
    elif len(description) > DESCRIPTION_MAX:
        violations.append(
            Violation(
                "description_too_long",
                f"SEO-описание длиннее {DESCRIPTION_MAX} символов ({len(description)}) — сократите",
            )
        )

    commercial = draft.commercial_block.strip()
    if not commercial:
        violations.append(Violation("commercial_missing", "Не заполнен commercial-блок"))
    elif not COMMERCIAL_MIN <= len(commercial) <= COMMERCIAL_MAX:
        violations.append(
            Violation(
                "commercial_length",
                f"Commercial-блок должен быть {COMMERCIAL_MIN}–{COMMERCIAL_MAX} символов "
                f"(сейчас {len(commercial)})",
            )
        )

    queries = [query for query in draft.search_queries if query.strip()]
    if not QUERIES_MIN <= len(queries) <= QUERIES_MAX:
        violations.append(
            Violation(
                "queries_count",
                f"Поисковых запросов должно быть {QUERIES_MIN}–{QUERIES_MAX} (сейчас {len(queries)})",
            )
        )
    if any(len(query) > MAX_QUERY_LENGTH for query in queries):
        violations.append(
            Violation("query_too_long", f"Поисковый запрос длиннее {MAX_QUERY_LENGTH} символов")
        )

    if len(draft.specs) < MIN_SPECS:
        violations.append(
            Violation(
                "specs_few",
                f"Характеристик меньше {MIN_SPECS} ({len(draft.specs)}): "
                "добавьте подтверждённые входными данными",
                severity="warning",
            )
        )
    elif len(draft.specs) > MAX_SPECS:
        violations.append(
            Violation(
                "specs_many",
                f"Характеристик больше {MAX_SPECS} ({len(draft.specs)}): оставьте главные",
                severity="warning",
            )
        )


def _check_language(draft: CardDraft, violations: list[Violation]) -> None:
    """Тексты карточки должны быть на русском."""
    if draft.seo_description and not re.search(r"[а-яА-ЯёЁ]", draft.seo_description):
        violations.append(
            Violation("wrong_language", "SEO-описание должно быть на русском языке")
        )


# ---------------------------------------------------------------------- утилиты


def _dedupe_queries(queries: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for query in queries:
        text = " ".join(query.split())
        if not text or len(text) > MAX_QUERY_LENGTH:
            continue
        key = lower(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _trim_title(title: str) -> str:
    text = title.strip()
    if len(text) <= MAX_TITLE_LENGTH:
        return text
    cut = text[:MAX_TITLE_LENGTH]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:—-")


def _trim_to(text: str, limit: int, min_keep: int) -> str:
    """Обрезает текст до лимита по границе предложения, если это возможно."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for separator in (". ", "! ", "? "):
        position = cut.rfind(separator)
        if position > min_keep * 0.7:
            return cut[: position + 1].strip()
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:—-")


def _merge_gaps(model_gaps: Sequence[str], gaps: Sequence[Gap]) -> list[str]:
    """Объединяет пропуски, найденные моделью и кодом, без дублей."""
    result: list[str] = []
    seen: set[str] = set()
    for item in model_gaps:
        text = item.strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            result.append(text)
    for gap in gaps:
        if gap.label.lower() not in seen:
            seen.add(gap.label.lower())
            result.append(gap.label)
    return result
