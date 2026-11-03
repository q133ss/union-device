"""Нормализация текста.

Единые правила приведения строк нужны в двух местах:
* проверка входных данных и поиск пропущенных характеристик;
* детерминированная проверка карточки на выдуманные значения.
"""

from __future__ import annotations

import re

# Разные тире, кавычки и неразрывные пробелы приводим к обычным символам.
_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "−": "-",
        "\u2011": "-",
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
    }
)

_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_NON_WORD = re.compile(r"[^a-zа-я0-9]+")
_CYRILLIC = re.compile(r"[а-яё]")


def clean(value: object) -> str:
    """Убирает лишние пробелы и приводит типографские символы к ASCII-аналогам."""
    return _WHITESPACE.sub(" ", str(value).translate(_TRANSLATION)).strip()


def lower(value: object) -> str:
    """Регистронезависимая форма с заменой «ё» на «е»."""
    return clean(value).casefold().replace("ё", "е")


def squeeze(value: object) -> str:
    """Оставляет только буквы и цифры.

    Позволяет находить значения независимо от пробелов и пунктуации:
    ``G1/2``, ``G 1/2`` и ``g1/2`` дают одну и ту же строку.
    """
    return _NON_WORD.sub("", lower(value))


def numbers(value: object) -> set[str]:
    """Возвращает числовые токены в каноническом виде: ``1,5`` и ``1.5`` → ``1.5``."""
    tokens: set[str] = set()
    for raw in _NUMBER.findall(clean(value)):
        token = raw.replace(",", ".")
        if token.endswith(".0"):
            token = token[:-2]
        tokens.add(token)
    return tokens


def has_cyrillic(value: str) -> bool:
    """True, если в строке есть кириллица (нужно для проверки языка результата)."""
    return bool(_CYRILLIC.search(lower(value)))
