"""Разбор YML-фида (Яндекс.Маркет) в JSON товаров для сервиса.

Фид «Союз-Прибор» — около 37 МБ и ~48 000 предложений, поэтому XML читается
потоково (``iterparse``), без загрузки всего документа в память.

Примеры:

    # посмотреть, что есть по модели
    python scripts/from_yml_feed.py --feed feed.xml --match "ТМ-510" --limit 3 --print

    # выгрузить товар и дополнить существующий JSON данными из фида
    python scripts/from_yml_feed.py --feed feed.xml --match "ТМ-510" --limit 1 \
        --out data/products --merge

В фиде нет описаний товаров, поэтому ``description`` заполняется отдельно —
фрагментом техпаспорта или текстом со страницы товара.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")

# Транслитерация для идентификаторов: «ТМ-510Р» → «tm-510r».
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


@dataclass
class Offer:
    """Одно предложение (конкретный SKU) из фида."""

    id: str
    name: str
    type_prefix: str = ""
    url: str = ""
    category_id: str = ""
    vendor: str = ""
    vendor_code: str = ""
    price: str = ""
    currency: str = "RUR"
    available: bool = True
    picture: str = ""
    params: dict[str, str] = field(default_factory=dict)


def iter_catalog(path: Path) -> tuple[dict[str, tuple[str, str]], list[Offer]]:
    """Читает фид один раз: возвращает дерево категорий и все предложения."""
    categories: dict[str, tuple[str, str]] = {}
    offers: list[Offer] = []

    for _, element in ElementTree.iterparse(str(path), events=("end",)):
        if element.tag == "category" and element.get("id"):
            categories[element.get("id")] = (
                element.get("parentId") or "",
                (element.text or "").strip(),
            )
            element.clear()
        elif element.tag == "offer" and element.get("id"):
            offers.append(parse_offer(element))
            element.clear()

    return categories, offers


def parse_offer(element: ElementTree.Element) -> Offer:
    """Превращает XML-узел предложения в :class:`Offer`."""
    params: dict[str, str] = {}
    for param in element.findall("param"):
        name = (param.get("name") or "").strip()
        value = (param.text or "").strip()
        if name and value:
            params[name] = value

    price_node = element.find("price")
    return Offer(
        id=element.get("id") or "",
        name=_text(element, "name"),
        type_prefix=_text(element, "typePrefix"),
        url=_text(element, "url"),
        category_id=_text(element, "categoryId"),
        vendor=_text(element, "vendor"),
        vendor_code=_text(element, "vendorCode"),
        price=(price_node.text or "").strip() if price_node is not None else "",
        currency=_text(element, "currencyId") or "RUR",
        available=(element.get("available") or "true").lower() == "true",
        picture=_text(element, "picture"),
        params=params,
    )


def category_path(categories: dict[str, tuple[str, str]], category_id: str) -> str:
    """Собирает путь «раздел / подраздел» по цепочке parentId."""
    parts: list[str] = []
    seen: set[str] = set()
    current = category_id
    while current and current in categories and current not in seen:
        seen.add(current)
        parent, name = categories[current]
        if name:
            parts.append(name)
        current = parent
    return " / ".join(reversed(parts))


def slugify(value: str) -> str:
    """Идентификатор товара: «ТМ-510Р.00(0-2,5 МПа)G1/2» → «tm-510r-00-0-2-5-mpa-g1-2»."""
    latin = "".join(_TRANSLIT.get(char, char) for char in value.lower())
    latin = latin.replace("мпа", "mpa").replace("кпа", "kpa").replace("кгс", "kgs")
    slug = SLUG_SEPARATORS.sub("-", latin).strip("-")
    return slug or "offer"


def offer_to_product(offer: Offer, categories: dict[str, tuple[str, str]]) -> dict:
    """Собирает входной JSON сервиса из предложения фида."""
    title = " ".join(part for part in (offer.type_prefix, offer.name) if part).strip()
    if title:
        title = title[0].upper() + title[1:]

    specs = dict(offer.params)
    if offer.vendor_code and "Артикул" not in specs:
        specs["Артикул"] = offer.vendor_code

    return {
        "id": slugify(offer.name or offer.id),
        "name": title or offer.name,
        "manufacturer": offer.vendor or None,
        "category": category_path(categories, offer.category_id) or None,
        "source_url": offer.url or None,
        "specs": specs,
        "description": None,
        "feed": {
            "offer_id": offer.id,
            "offer_name": offer.name,
            "price": offer.price,
            "currency": offer.currency,
            "available": offer.available,
        },
    }


def merge_products(existing: dict, feed_product: dict) -> dict:
    """Дополняет существующий JSON данными фида.

    Приоритет: производитель, категория и параметры SKU — из фида (это
    первоисточник магазина), остальные поля и характеристики — из существующего
    файла, чтобы не потерять данные, которых в фиде нет (описание, развёрнутые
    технические характеристики).
    """
    merged = dict(existing)
    merged["manufacturer"] = feed_product.get("manufacturer") or existing.get("manufacturer")
    merged["category"] = feed_product.get("category") or existing.get("category")
    merged["source_url"] = feed_product.get("source_url") or existing.get("source_url")
    merged.setdefault("id", feed_product.get("id"))

    specs = dict(existing.get("specs") or {})
    specs.update(feed_product.get("specs") or {})
    merged["specs"] = specs
    merged["feed"] = feed_product.get("feed", {})
    merged.setdefault("description", existing.get("description"))
    return merged


def select(
    offers: list[Offer],
    matches: list[str],
    offer_ids: list[str],
    limit: int | None,
) -> list[Offer]:
    """Отбирает предложения по подстроке названия и/или id."""
    selected: list[Offer] = []
    for offer in offers:
        if offer_ids and offer.id not in offer_ids:
            continue
        if matches:
            haystack = f"{offer.name} {offer.type_prefix}".lower()
            if not any(match.lower() in haystack for match in matches):
                continue
        selected.append(offer)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертер YML-фида в JSON товаров сервиса",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feed", type=Path, required=True, help="путь к feed.xml")
    parser.add_argument("--match", action="append", default=[], help="подстрока названия (можно несколько)")
    parser.add_argument("--offer-id", action="append", default=[], help="id предложения (можно несколько)")
    parser.add_argument("--limit", type=int, default=None, help="сколько предложений взять")
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить JSON товаров")
    parser.add_argument("--merge", action="store_true", help="дополнить существующие JSON в --out")
    parser.add_argument("--print", dest="print_only", action="store_true", help="только показать JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.feed.is_file():
        print(f"Файл фида не найден: {args.feed}", file=sys.stderr)
        return 2
    if not args.match and not args.offer_id:
        print("Укажите --match или --offer-id, чтобы выбрать товары", file=sys.stderr)
        return 2

    categories, offers = iter_catalog(args.feed)
    selected = select(offers, args.match, args.offer_id, args.limit)
    if not selected:
        print("Ничего не найдено по заданным условиям", file=sys.stderr)
        return 1

    products = [offer_to_product(offer, categories) for offer in selected]
    print(f"Категорий: {len(categories)}, предложений в фиде: {len(offers)}, выбрано: {len(products)}")

    if args.print_only or not args.out:
        for product in products:
            print(json.dumps(product, ensure_ascii=False, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for product in products:
        target = args.out / f"{product['id']}.json"
        if args.merge and target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            product = merge_products(existing, product)
        target.write_text(
            json.dumps(product, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  → {target}")
    return 0


def _text(element: ElementTree.Element, tag: str) -> str:
    node = element.find(tag)
    return (node.text or "").strip() if node is not None else ""


if __name__ == "__main__":
    raise SystemExit(main())
