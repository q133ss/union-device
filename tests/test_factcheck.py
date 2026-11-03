"""Тесты детерминированной проверки карточки на выдуманные данные."""

from __future__ import annotations

import pytest

from app.factcheck import check, sanitize
from app.models import CardDraft
from app.validation import parse_product

TM510 = {
    "name": "Манометр показывающий ТМ-510",
    "manufacturer": "Манотомь",
    "specs": {
        "Диапазон измерений (МПа)": "от 0-1 кгс/см² до 0-100 МПа, зависит от исполнения",
        "Резьба присоединения": "G1/2; М20×1,5",
        "Расположение штуцера": "радиальное; осевое",
    },
    "description": None,
}

PR200 = {
    "name": "Программируемое реле с дисплеем ОВЕН ПР200",
    "manufacturer": "ОВЕН",
    "specs": {
        "Корпус": "7 DIN",
        "Интерфейсы связи": "до 2 × RS-485, протоколы Modbus RTU/ASCII",
        "Дискретные входы": "8",
        "Дисплей": "текстовый монохромный ЖКИ с подсветкой, 2×16 символов",
    },
    "description": "Применяется для локальных задач автоматизации.",
}

LONG_TEXT = (
    "Манометр ТМ-510 предназначен для контроля избыточного давления в трубопроводах. "
    "Прибор применяется на объектах водоснабжения и в котельных. "
) * 8


def build(product: dict, **overrides) -> tuple[CardDraft, object, object]:
    analysis = parse_product(product)
    payload = {
        "seo_title": "Манометр показывающий ТМ-510 Манотомь, G1/2",
        "seo_description": LONG_TEXT,
        "specs": [{"name": "Резьба присоединения", "value": "G1/2"}],
        "search_queries": ["купить манометр тм-510", "манометр тм-510 характеристики", "манометр g1/2"],
        "commercial_block": "Подходит для котельных и водоканалов. " * 10,
    }
    payload.update(overrides)
    return CardDraft.model_validate(payload), analysis.product, analysis.gaps


def test_hallucinated_spec_is_rejected():
    draft, product, gaps = build(
        TM510, specs=[{"name": "Класс точности", "value": "1,5"}]
    )
    report = check(draft, product, gaps)
    assert 0 in report.invalid_specs
    assert any("Класс точности" in error.message for error in report.errors)


def test_value_from_other_field_is_not_accepted():
    """«1,5» есть во входе как часть резьбы М20×1,5, но не как класс точности."""
    draft, product, gaps = build(TM510, specs=[{"name": "Класс точности", "value": "1,5"}])
    report = check(draft, product, gaps)
    assert report.invalid_specs == (0,)


def test_renamed_but_supported_spec_is_accepted():
    draft, product, gaps = build(
        PR200,
        seo_title="Программируемое реле ОВЕН ПР200, 7 DIN",
        specs=[
            {"name": "Интерфейсы", "value": "до 2 × RS-485, протоколы Modbus RTU/ASCII"},
            {"name": "Корпус", "value": "7 DIN"},
            {"name": "Дискретные входы", "value": "8"},
        ],
    )
    report = check(draft, product, gaps)
    assert report.invalid_specs == ()


def test_data_gap_marker_is_accepted():
    draft, product, gaps = build(
        TM510, specs=[{"name": "Класс точности", "value": "данных нет"}]
    )
    report = check(draft, product, gaps)
    assert report.invalid_specs == ()
    assert not any(error.code == "gap_filled" for error in report.errors)


def test_invented_number_in_description_is_reported():
    draft, product, gaps = build(
        TM510,
        seo_description=LONG_TEXT + " Прибор выпускается с 1998 года.",
    )
    report = check(draft, product, gaps)
    assert any(error.code == "number_not_in_source" for error in report.errors)


def test_volumes_are_enforced():
    draft, product, gaps = build(
        TM510,
        seo_title="М" * 80,
        search_queries=["один запрос"],
        commercial_block="коротко",
    )
    report = check(draft, product, gaps)
    codes = {error.code for error in report.errors}
    assert {"title_too_long", "queries_count", "commercial_length"} <= codes


def test_sanitize_removes_unconfirmed_specs_and_merges_gaps():
    draft, product, gaps = build(
        TM510,
        specs=[
            {"name": "Резьба присоединения", "value": "G1/2"},
            {"name": "Диаметр корпуса", "value": "160 мм"},
        ],
    )
    report = check(draft, product, gaps)
    clean = sanitize(draft, report, gaps)

    assert [spec.name for spec in clean.specs] == ["Резьба присоединения"]
    assert any("Удалены характеристики" in warning for warning in clean.warnings)
    assert any("Класс точности" in gap for gap in clean.data_gaps)


def test_sanitize_trims_title_and_dedupes_queries():
    draft, product, gaps = build(
        TM510,
        seo_title="Манометр показывающий ТМ-510 Манотомь радиальный штуцер G1/2 100 МПа",
        search_queries=["купить манометр", "купить манометр", "манометр тм-510"],
    )
    clean = sanitize(draft, check(draft, product, gaps), gaps)

    assert len(clean.seo_title) <= 70
    assert len(clean.search_queries) == 2


def test_sanitize_trims_long_commercial_block():
    draft, product, gaps = build(TM510, commercial_block="Подходит для котельных. " * 40)
    assert len(draft.commercial_block) > 600
    clean = sanitize(draft, check(draft, product, gaps), gaps)

    assert len(clean.commercial_block) <= 600
    assert any("Commercial-блок сокращён" in warning for warning in clean.warnings)


def test_report_serializes_to_dict():
    draft, product, gaps = build(TM510)
    payload = check(draft, product, gaps).as_dict()

    assert set(payload) == {"passed", "errors", "warnings", "stats"}
    assert isinstance(payload["passed"], bool)


@pytest.mark.parametrize(
    "value",
    ["100 МПа", "М20×1,5", "радиальное; осевое"],
)
def test_real_values_are_supported(value: str):
    draft, product, gaps = build(TM510, specs=[{"name": "Диапазон измерений", "value": value}])
    report = check(draft, product, gaps)
    assert report.invalid_specs == ()
