"""Модели данных сервиса.

Входная модель (``ProductInput``) описывает то, что приходит из формы,
JSON-файла или фида. Она намеренно терпима к «грязным» данным: числа,
списки и пустые значения нормализуются, а не приводят к падению сервиса.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.normalize import clean

MAX_NAME = 300
MAX_MANUFACTURER = 200
MAX_SPEC_KEY = 120
MAX_SPEC_VALUE = 400
MAX_SPECS = 40
MAX_DESCRIPTION = 8000


class ProductInput(BaseModel):
    """Данные прибора, по которым строится карточка товара."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    id: str | None = Field(default=None, description="Идентификатор товара (sku/slug)")
    name: str = Field(..., description="Название товара")
    manufacturer: str | None = Field(default=None, description="Производитель")
    category: str | None = Field(default=None, description="Категория / раздел каталога")
    source_url: str | None = Field(default=None, description="Ссылка на источник данных")
    specs: dict[str, str] = Field(
        default_factory=dict, description="Характеристики «название → значение»"
    )
    description: str | None = Field(
        default=None, description="Текстовое описание или фрагмент техпаспорта"
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = clean(value)
        if len(value) < 2:
            raise ValueError("название товара обязательно и должно быть не короче 2 символов")
        if len(value) > MAX_NAME:
            raise ValueError(f"название длиннее {MAX_NAME} символов")
        return value

    @field_validator("manufacturer", "category", "source_url", "id", mode="before")
    @classmethod
    def _optional_str(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = clean(value)
        return text or None

    @field_validator("manufacturer")
    @classmethod
    def _validate_manufacturer(cls, value: str | None) -> str | None:
        if value is not None and len(value) > MAX_MANUFACTURER:
            raise ValueError(f"производитель длиннее {MAX_MANUFACTURER} символов")
        return value

    @field_validator("description", mode="before")
    @classmethod
    def _optional_description(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("specs", mode="before")
    @classmethod
    def _normalize_specs(cls, value: Any) -> dict[str, str]:
        """Числа, списки и None приводим к строкам; пустые значения отбрасываем."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("характеристики должны быть объектом «название → значение»")
        if len(value) > MAX_SPECS:
            raise ValueError(f"слишком много характеристик: {len(value)}, максимум {MAX_SPECS}")

        specs: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = clean(raw_key)
            if not key:
                continue
            if len(key) > MAX_SPEC_KEY:
                key = key[:MAX_SPEC_KEY]
            text = _spec_value_to_text(raw_value)
            if not text:
                continue
            if len(text) > MAX_SPEC_VALUE:
                text = text[:MAX_SPEC_VALUE]
            specs.setdefault(key, text)
        return specs

    @property
    def display_name(self) -> str:
        """Название без бренда — удобно для заголовка карточки."""
        return self.name

    @property
    def flat_text(self) -> str:
        """Все входные данные одной строкой — база для проверки на выдуманные значения."""
        chunks = [self.name, self.manufacturer or "", self.category or "", self.description or ""]
        chunks.extend(f"{key}: {value}" for key, value in self.specs.items())
        return "\n".join(chunk for chunk in chunks if chunk)


def _spec_value_to_text(value: Any) -> str:
    """Приводит значение характеристики к непустой строке."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return clean(value)
    if isinstance(value, (list, tuple)):
        parts = [_spec_value_to_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    return clean(value)
