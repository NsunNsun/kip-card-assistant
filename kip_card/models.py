"""Модели данных: вход, ответ модели и итоговая карточка."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NO_DATA = "данных нет"

Level = Literal["error", "warning", "info"]


class Issue(BaseModel):
    """Замечание для контент-менеджера: чего не хватает или что проверить."""

    level: Level
    field: str
    message: str


class ProductInput(BaseModel):
    """Нормализованные входные данные о товаре."""

    name: str
    manufacturer: str = ""
    product_type: str = ""
    category: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    identifiers: dict[str, str] = Field(default_factory=dict)
    source_url: str = ""

    def facts_for_llm(self) -> dict:
        """Только то, что модели можно использовать как факты (без идентификаторов и URL)."""
        data = {
            "название": self.name,
            "тип прибора": self.product_type,
            "производитель": self.manufacturer,
            "категория каталога": self.category,
            "характеристики": self.specs,
            "описание": self.description,
        }
        return {k: v for k, v in data.items() if v}


class Commercial(BaseModel):
    summary: str = Field(description="1–2 предложения: что это за прибор и зачем он покупателю")
    who_for: list[str] = Field(description="Кому подходит прибор: отрасли, службы, специалисты")
    tasks: list[str] = Field(description="Какие задачи решает прибор")


class LLMCopy(BaseModel):
    """То, что пишет модель. Характеристики сюда не входят — их собирает код."""

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial: Commercial
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Чего не хватает во входных данных для хорошей карточки (по мнению модели)",
    )


class SpecRow(BaseModel):
    name: str
    value: str
    source: Literal["input", "missing"]


class Meta(BaseModel):
    model: str
    generated_at: str
    attempts: int
    response_format: str
    prompt_version: int = 1


class ProductCard(BaseModel):
    status: Literal["ok", "needs_review"]
    product: dict[str, str]
    seo_title: str
    seo_description: str
    key_specs: list[SpecRow]
    search_queries: list[str]
    commercial: Commercial
    missing_fields: list[str]
    issues: list[Issue]
    meta: Meta
