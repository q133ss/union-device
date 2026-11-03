"""Проверка входных данных и поиск недостающих характеристик.

Модуль решает две задачи:
1. Не падать на «грязном» входе: пустые поля, числа вместо строк, мусорный JSON —
   вернуть понятную ошибку на русском вместо 500.
2. До вызова LLM понять, каких характеристик не хватает, — чтобы промпт явно
   требовал «данных нет» вместо выдуманных значений.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, NamedTuple

from pydantic import ValidationError

from app.models import MAX_DESCRIPTION, ProductInput
from app.normalize import lower


@dataclass(frozen=True)
class FieldIssue:
    """Одна проблема входных данных."""

    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message}


class InputError(ValueError):
    """Входные данные непригодны для генерации карточки."""

    def __init__(self, issues: Iterable[FieldIssue]) -> None:
        self.issues = list(issues)
        super().__init__("; ".join(f"{issue.field}: {issue.message}" for issue in self.issues))


class SpecGroup(NamedTuple):
    """Группа характеристик, наличие которой проверяется эвристикой."""

    key: str
    label: str
    hint: str
    patterns: tuple[str, ...]


# Чек-лист типовых характеристик КИП. Если ни один шаблон не найден во входных
# данных — группа считается отсутствующей и попадает в data_gaps.
SPEC_GROUPS: tuple[SpecGroup, ...] = (
    SpecGroup(
        "диапазон измерений",
        "Диапазон измерений",
        "укажите пределы измерений с единицами (например, 0–1,6 МПа)",
        (r"диапазон", r"предел измерен", r"верхний предел", r"range"),
    ),
    SpecGroup(
        "класс точности",
        "Класс точности / погрешность",
        "класс точности, предел допускаемой погрешности",
        (r"класс точности", r"точност", r"погрешност", r"accuracy"),
    ),
    SpecGroup(
        "питание",
        "Питание",
        "напряжение и род тока, потребляемая мощность",
        (r"питани", r"напряжени", r"power", r"потребляем"),
    ),
    SpecGroup(
        "интерфейсы",
        "Интерфейсы связи",
        "RS-485/Modbus, USB, Ethernet, дискретные входы/выходы",
        (r"интерфейс", r"rs-?\s?485", r"modbus", r"\busb\b", r"ethernet", r"выход"),
    ),
    SpecGroup(
        "корпус",
        "Корпус / степень защиты",
        "тип корпуса, габариты, степень защиты IP",
        (r"корпус", r"\bdin\b", r"степен[ьи] защиты", r"\bip\s?\d", r"габарит"),
    ),
    SpecGroup(
        "присоединение",
        "Присоединение",
        "резьба, тип штуцера, способ монтажа",
        (r"резьб", r"присоединен", r"штуцер", r"\bg\s?1/2", r"\bм20"),
    ),
    SpecGroup(
        "индикация",
        "Индикация / дисплей",
        "тип дисплея и его возможности",
        (r"дисплей", r"индикац", r"жк", r"экран", r"display"),
    ),
)


@dataclass(frozen=True)
class Gap:
    """Недостающая характеристика."""

    key: str
    label: str
    hint: str

    def as_prompt_line(self) -> str:
        return f"{self.label} — {self.hint}"


@dataclass(frozen=True)
class Analysis:
    """Результат разбора входа: нормализованный товар, пропуски, замечания."""

    product: ProductInput
    gaps: tuple[Gap, ...]
    notes: tuple[str, ...]


def parse_product(raw: Any) -> Analysis:
    """Проверяет сырые данные и возвращает разбор с пропусками."""
    if raw is None:
        raise InputError([FieldIssue("body", "пустое тело запроса: ожидался JSON-объект товара")])
    if isinstance(raw, str):
        raise InputError(
            [FieldIssue("body", "ожидался JSON-объект товара, а не строка — проверьте тело запроса")]
        )
    if not isinstance(raw, dict):
        raise InputError([FieldIssue("body", "ожидался JSON-объект товара")])

    data = dict(raw)
    notes: list[str] = []

    description = data.get("description")
    if isinstance(description, str) and len(description) > MAX_DESCRIPTION:
        data["description"] = description[:MAX_DESCRIPTION]
        notes.append(
            f"Описание обрезано до {MAX_DESCRIPTION} символов (было {len(description)}): "
            "полный текст не влияет на карточку, но экономит токены"
        )

    try:
        product = ProductInput.model_validate(data)
    except ValidationError as exc:
        raise InputError(_issues_from_pydantic(exc)) from exc

    if not product.manufacturer and not product.specs and not product.description:
        raise InputError(
            [
                FieldIssue(
                    "payload",
                    "недостаточно данных: нужен хотя бы производитель, характеристики "
                    "или описание — иначе карточку не из чего собрать",
                )
            ]
        )

    return Analysis(product=product, gaps=tuple(detect_gaps(product)), notes=tuple(notes))


def detect_gaps(product: ProductInput) -> list[Gap]:
    """Находит отсутствующие группы характеристик и обязательные поля."""
    haystack = lower(product.flat_text)
    gaps: list[Gap] = []

    if not product.description:
        gaps.append(
            Gap(
                "описание",
                "Описание товара",
                "нужен фрагмент техпаспорта или иное описание назначения прибора",
            )
        )
    if not product.manufacturer:
        gaps.append(Gap("производитель", "Производитель", "укажите производителя прибора"))

    for group in SPEC_GROUPS:
        if not any(re.search(pattern, haystack) for pattern in group.patterns):
            gaps.append(Gap(group.key, group.label, group.hint))

    return gaps


def _issues_from_pydantic(exc: ValidationError) -> list[FieldIssue]:
    """Переводит ошибки pydantic в плоский список понятных сообщений."""
    issues: list[FieldIssue] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "payload"
        message = str(error.get("msg", "некорректное значение"))
        message = message.removeprefix("Value error, ").strip()
        issues.append(FieldIssue(location, message))
    return issues
