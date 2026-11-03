"""Тесты проверки входных данных и определения пропусков."""

from __future__ import annotations

import pytest

from app.validation import InputError, detect_gaps, parse_product


def test_valid_product_is_normalized():
    analysis = parse_product(
        {
            "name": "  Манометр ТМ-510  ",
            "manufacturer": "Манотомь",
            "specs": {"Диапазон": "0–100 МПа"},
        }
    )
    assert analysis.product.name == "Манометр ТМ-510"
    assert analysis.product.manufacturer == "Манотомь"
    assert analysis.product.specs == {"Диапазон": "0-100 МПа"}


def test_specs_values_are_coerced_to_text():
    analysis = parse_product(
        {
            "name": "Прибор",
            "manufacturer": "Завод",
            "specs": {"Диапазон": 5, "Наличие": True, "Пусто": None, "Комплект": ["кабель", "паспорт"]},
        }
    )
    assert analysis.product.specs == {
        "Диапазон": "5",
        "Наличие": "да",
        "Комплект": "кабель, паспорт",
    }


def test_empty_name_is_rejected():
    with pytest.raises(InputError) as error:
        parse_product({"name": "  "})
    assert error.value.issues[0].field == "name"
    assert "название" in error.value.issues[0].message.lower()


def test_non_object_body_is_rejected():
    with pytest.raises(InputError) as error:
        parse_product(["товар"])
    assert error.value.issues[0].field == "body"


def test_product_without_any_data_is_rejected():
    with pytest.raises(InputError) as error:
        parse_product({"name": "Манометр"})
    assert error.value.issues[0].field == "payload"


def test_wrong_specs_type_is_rejected():
    with pytest.raises(InputError) as error:
        parse_product({"name": "Манометр", "specs": ["диапазон", "0-1 МПа"]})
    assert error.value.issues[0].field == "specs"


def test_long_description_is_truncated_with_note():
    analysis = parse_product(
        {"name": "Манометр", "manufacturer": "Завод", "description": "а" * 9000}
    )
    assert len(analysis.product.description or "") == 8000
    assert analysis.notes and "обрезано" in analysis.notes[0]


def test_gaps_detected_for_sparse_product():
    analysis = parse_product(
        {
            "name": "Манометр показывающий ТМ-510",
            "manufacturer": "Манотомь",
            "specs": {"Резьба присоединения": "G1/2"},
        }
    )
    labels = [gap.label for gap in analysis.gaps]
    assert "Описание товара" in labels
    assert "Класс точности / погрешность" in labels
    assert "Питание" in labels


def test_gaps_not_reported_when_data_complete():
    product = parse_product(
        {
            "name": "Прибор",
            "manufacturer": "Завод",
            "specs": {
                "Диапазон измерений": "0–1,6 МПа",
                "Класс точности": "1,5",
                "Питание": "24 В",
                "Интерфейс RS-485": "Modbus RTU",
                "Корпус": "IP65, 100 мм",
                "Резьба присоединения": "G1/2",
                "Дисплей": "ЖКИ",
            },
            "description": "Прибор для измерения давления.",
        }
    ).product
    assert detect_gaps(product) == []


def test_missing_manufacturer_is_a_gap():
    analysis = parse_product({"name": "Прибор", "specs": {"Питание": "24 В"}})
    assert any(gap.key == "производитель" for gap in analysis.gaps)
