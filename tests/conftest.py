"""Общие фикстуры для тестов."""

from __future__ import annotations

import pytest

PRODUCT = {
    "name": "Манометр показывающий ТМ-510",
    "manufacturer": "Манотомь",
    "source_url": "https://www.souz-pribor.ru/product/tm-510/",
    "specs": {
        "Диапазон измерений (МПа)": "0-100 МПа",
        "Резьба присоединения": "G1/2",
        "Расположение штуцера": "радиальное",
        "Исполнение": "стандартное",
    },
    "description": "Манометр для измерения избыточного давления.",
}

GOOD_DRAFT = {
    "seo_title": "Манометр показывающий ТМ-510 Манотомь, G1/2",
    "seo_description": (
        "Манометр показывающий ТМ-510 применяется для контроля избыточного давления "
        "в трубопроводах и технологических линиях. Прибор рассчитан на диапазон до "
        "100 МПа и присоединяется через резьбу G1/2. "
    )
    * 6,
    "specs": [
        {"name": "Диапазон измерений", "value": "0-100 МПа"},
        {"name": "Резьба присоединения", "value": "G1/2"},
        {"name": "Расположение штуцера", "value": "радиальное"},
        {"name": "Исполнение", "value": "стандартное"},
    ],
    "search_queries": [
        "купить манометр тм-510",
        "манометр тм-510 g1/2",
        "манометр на 100 мпа",
    ],
    "commercial_block": (
        "Подходит для котельных, водоканалов и производственных участков, где нужен "
        "контроль давления по месту. "
    )
    * 3,
    "data_gaps": [],
    "warnings": [],
}


@pytest.fixture
def product_payload() -> dict:
    return dict(PRODUCT)


@pytest.fixture
def good_draft() -> dict:
    return {key: (value if not isinstance(value, list) else list(value)) for key, value in GOOD_DRAFT.items()}
