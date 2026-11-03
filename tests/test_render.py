"""Тесты рендеринга карточки в Markdown и JSON."""

from __future__ import annotations

import json

from app.models import ProductCard, SpecItem
from app.render import to_json, to_markdown


def build_card(**overrides) -> ProductCard:
    payload = {
        "name": "Манометр показывающий ТМ-510",
        "manufacturer": "Манотомь",
        "source_url": "https://www.souz-pribor.ru/product/tm-510/",
        "seo_title": "Манометр показывающий ТМ-510 Манотомь, G1/2",
        "seo_description": "Описание прибора. " * 50,
        "specs": [SpecItem(name="Резьба присоединения", value="G1/2")],
        "search_queries": ["купить манометр тм-510"],
        "commercial_block": "Кому подходит: котельные.",
        "data_gaps": ["Класс точности"],
        "warnings": [],
        "fact_check": {"passed": True, "errors": [], "warnings": [], "stats": {}},
        "meta": {
            "llm": {"model": "test-model", "calls": 1, "total_tokens": 1234, "latency_ms": 1500},
            "generated_at": "2026-11-03T10:00:00+00:00",
        },
    }
    payload.update(overrides)
    return ProductCard(**payload)


def test_markdown_contains_all_sections():
    markdown = to_markdown(build_card())

    assert markdown.startswith("# Манометр показывающий ТМ-510 Манотомь, G1/2")
    for section in ("## SEO-заголовок", "## SEO-описание", "## Характеристики",
                    "## Поисковые запросы", "## Commercial-блок", "## Чего не хватает"):
        assert section in markdown
    assert "| Резьба присоединения | G1/2 |" in markdown
    assert "- Класс точности — данных нет" in markdown
    assert "проверка достоверности пройдена" in markdown


def test_markdown_marks_missing_manufacturer():
    markdown = to_markdown(build_card(manufacturer=""))
    assert "**Производитель:** данных нет" in markdown


def test_markdown_marks_failed_check():
    markdown = to_markdown(build_card(fact_check={"passed": False, "errors": [], "warnings": []}))
    assert "есть замечания проверки достоверности" in markdown


def test_markdown_escapes_table_pipes():
    card = build_card(specs=[SpecItem(name="Питание", value="24 В | 220 В")])
    assert "24 В \\| 220 В" in to_markdown(card)


def test_json_is_valid_and_round_trips():
    card = build_card()
    parsed = json.loads(to_json(card))

    assert parsed["seo_title"] == card.seo_title
    assert parsed["specs"][0]["value"] == "G1/2"
    assert parsed["fact_check"]["passed"] is True
