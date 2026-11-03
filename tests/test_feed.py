"""Тесты конвертера YML-фида."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from from_yml_feed import (  # noqa: E402
    category_path,
    iter_catalog,
    merge_products,
    offer_to_product,
    select,
    slugify,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "feed_sample.xml"


@pytest.fixture(scope="module")
def catalog():
    return iter_catalog(FIXTURE)


def test_offers_are_parsed(catalog):
    _, offers = catalog
    assert len(offers) == 2
    tm = offers[0]
    assert tm.id == "191953"
    assert tm.vendor == "РОСМА"
    assert tm.price == "1045"
    assert tm.available is True
    assert tm.params == {"Код товара": "77321"}


def test_category_path_is_resolved(catalog):
    categories, _ = catalog
    assert category_path(categories, "343") == "Оборудование для промышленной автоматизации / Программируемые реле"
    assert category_path(categories, "369") == "Теплотехнические приборы"


def test_offer_to_product(catalog):
    categories, offers = catalog
    product = offer_to_product(offers[1], categories)

    assert product["name"] == "Программируемое реле с дисплеем ПР200-220.1.1.0"
    assert product["manufacturer"] == "ОВЕН"
    assert product["category"].endswith("Программируемые реле")
    assert product["specs"]["Артикул"] == "77510"
    assert product["feed"]["available"] is False
    assert product["description"] is None


def test_merge_keeps_existing_specs_and_takes_vendor_from_feed(catalog):
    categories, offers = catalog
    feed_product = offer_to_product(offers[0], categories)
    existing = {
        "id": "tm-510",
        "name": "Манометр показывающий ТМ-510",
        "manufacturer": "Манотомь",
        "specs": {"Резьба присоединения": "G1/2", "Код товара": "старый"},
        "description": "Фрагмент техпаспорта.",
    }

    merged = merge_products(existing, feed_product)

    assert merged["manufacturer"] == "РОСМА"
    assert merged["name"] == "Манометр показывающий ТМ-510"
    assert merged["description"] == "Фрагмент техпаспорта."
    assert merged["specs"]["Резьба присоединения"] == "G1/2"
    assert merged["specs"]["Код товара"] == "77321"
    assert merged["feed"]["offer_id"] == "191953"


def test_select_by_substring_and_limit(catalog):
    _, offers = catalog
    assert len(select(offers, ["ТМ-510"], [], None)) == 1
    assert len(select(offers, [], [], 1)) == 1
    assert select(offers, ["нет-такого"], [], None) == []


def test_slugify_transliterates_russian_units():
    assert slugify("ТМ-510Р.00(0-2,5 МПа)G1/2") == "tm-510r-00-0-2-5-mpa-g1-2"


def test_product_is_serializable(catalog):
    categories, offers = catalog
    payload = offer_to_product(offers[0], categories)
    assert json.loads(json.dumps(payload, ensure_ascii=False))["manufacturer"] == "РОСМА"


def test_converter_cli_writes_files(tmp_path, catalog):
    from from_yml_feed import main

    assert main(["--feed", str(FIXTURE), "--match", "ПР200", "--out", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["manufacturer"] == "ОВЕН"
