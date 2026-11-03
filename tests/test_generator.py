"""Тесты оркестрации генерации с подменённым LLM-клиентом."""

from __future__ import annotations

import asyncio

import pytest

from app.generator import generate_card
from app.llm import LLMError, LLMResult
from app.validation import parse_product

PRODUCT = {
    "name": "Манометр показывающий ТМ-510",
    "manufacturer": "Манотомь",
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
        "в трубопроводах и технологических линиях. "
    )
    * 9,
    "specs": [
        {"name": "Диапазон измерений", "value": "0-100 МПа"},
        {"name": "Резьба присоединения", "value": "G1/2"},
        {"name": "Расположение штуцера", "value": "радиальное"},
        {"name": "Исполнение", "value": "стандартное"},
    ],
    "search_queries": ["купить манометр тм-510", "манометр тм-510 g1/2", "манометр на 100 мпа"],
    "commercial_block": (
        "Подходит для котельных, водоканалов и производственных участков, "
        "где нужен контроль давления по месту. "
    )
    * 3,
    "data_gaps": [],
    "warnings": [],
}


class FakeClient:
    """Отдаёт заранее подготовленные ответы по очереди."""

    def __init__(self, drafts: list[dict]):
        self.drafts = drafts
        self.calls = 0
        self.prompts: list[str] = []

    async def complete_json(self, system: str, user: str, **kwargs):
        self.prompts.append(user)
        draft = self.drafts[min(self.calls, len(self.drafts) - 1)]
        self.calls += 1
        return draft, LLMResult(
            content="{}",
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=25,
        )


def run_generation(client: FakeClient):
    analysis = parse_product(PRODUCT)
    return asyncio.run(generate_card(analysis, client))


def test_clean_draft_needs_no_repair():
    client = FakeClient([GOOD_DRAFT])
    result = run_generation(client)

    assert client.calls == 1
    assert result.repair_attempts == 0
    assert result.card.fact_check["passed"] is True
    assert result.card.meta["llm"]["calls"] == 1


def test_hallucinated_spec_triggers_repair_request():
    bad = dict(GOOD_DRAFT)
    bad["specs"] = GOOD_DRAFT["specs"] + [{"name": "Класс точности", "value": "1,5"}]
    client = FakeClient([bad, GOOD_DRAFT])

    result = run_generation(client)

    assert client.calls == 2
    assert result.repair_attempts == 1
    assert "Класс точности" in client.prompts[1]
    assert all(spec.name != "Класс точности" for spec in result.card.specs)


def test_unfixable_hallucination_is_removed_and_reported():
    bad = dict(GOOD_DRAFT)
    bad["specs"] = GOOD_DRAFT["specs"] + [{"name": "Диаметр корпуса", "value": "160 мм"}]
    client = FakeClient([bad])

    result = run_generation(client)

    assert all(spec.name != "Диаметр корпуса" for spec in result.card.specs)
    assert any("Удалены характеристики" in warning for warning in result.card.warnings)
    assert result.card.meta["gaps_detected"]


def test_long_title_is_trimmed_to_limit():
    draft = dict(GOOD_DRAFT)
    draft["seo_title"] = (
        "Манометр показывающий ТМ-510 Манотомь, радиальный штуцер, резьба G1/2, 100 МПа"
    )
    assert len(draft["seo_title"]) > 70
    result = run_generation(FakeClient([draft]))

    assert len(result.card.seo_title) <= 70
    assert result.card.meta["repair_attempts"] == 1


def test_broken_schema_is_reported_as_llm_error():
    class BrokenClient:
        async def complete_json(self, system: str, user: str, **kwargs):
            return ["не объект"], LLMResult(content="[]", model="fake")

    analysis = parse_product(PRODUCT)
    with pytest.raises(LLMError) as error:
        asyncio.run(generate_card(analysis, BrokenClient()))

    assert error.value.code == "llm_invalid_json"


def test_card_contains_source_and_metadata():
    result = run_generation(FakeClient([GOOD_DRAFT]))
    card = result.card

    assert card.manufacturer == "Манотомь"
    assert card.meta["llm"]["model"] == "fake-model"
    assert card.meta["generated_at"].endswith("+00:00")
    assert card.fact_check["stats"]["specs_checked"] == 4
