"""Тесты сборки промптов."""

from __future__ import annotations

from app.prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from app.validation import parse_product

PRODUCT = {
    "id": "tm-510",
    "name": "Манометр показывающий ТМ-510",
    "manufacturer": "РОСМА",
    "category": "Теплотехнические приборы, газоанализаторы, счетчики / Манометры показывающие",
    "source_url": "https://www.souz-pribor.ru/product/tm-510/",
    "specs": {"Резьба присоединения": "G1/2"},
    "description": "Манометр для измерения избыточного давления.",
}


def test_system_prompt_sets_hard_rules():
    lowered = SYSTEM_PROMPT.lower()
    assert "данных нет" in lowered
    assert "не должно быть ни одного факта" in lowered
    assert "json" in lowered


def test_user_prompt_has_data_and_gaps():
    analysis = parse_product(PRODUCT)
    prompt = build_user_prompt(analysis.product, analysis.gaps)

    assert "Манометр показывающий ТМ-510" in prompt
    assert "РОСМА" in prompt
    assert "G1/2" in prompt
    assert "Класс точности" in prompt  # найденный пропуск подсказан модели


def test_catalog_section_is_not_sent_to_the_model():
    """Раздел каталога — навигация магазина, а не назначение прибора."""
    analysis = parse_product(PRODUCT)
    prompt = build_user_prompt(analysis.product, analysis.gaps)

    assert "газоанализатор" not in prompt.lower()
    assert "category" not in prompt


def test_repair_prompt_lists_violations():
    analysis = parse_product(PRODUCT)
    prompt = build_repair_prompt(analysis.product, {"seo_title": "x"}, ["выдуманное число 1998"])
    assert "1998" in prompt
    assert "РОСМА" in prompt
