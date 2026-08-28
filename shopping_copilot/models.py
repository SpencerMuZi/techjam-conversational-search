from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ProductDocument:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    price: float | None
    average_rating: float
    rating_number: int
    searchable_text: str


@dataclass(frozen=True)
class ParsedTurn:
    route_hint: str | None = None
    category: str | None = None
    slots: dict[str, list[str]] = field(default_factory=dict)
    no_preference: str | None = None
    is_override: bool = False


@dataclass(frozen=True)
class SearchContext:
    route: str
    category: str
    hard_slots: dict[str, tuple[str, ...]]
    soft_slots: dict[str, tuple[str, ...]]
    profile_tags: tuple[str, ...]
    current_message: str

    @property
    def hard_values(self) -> tuple[str, ...]:
        return tuple(value for values in self.hard_slots.values() for value in values)

    @property
    def soft_values(self) -> tuple[str, ...]:
        return tuple(value for values in self.soft_slots.values() for value in values)


class SemanticRetriever(Protocol):
    """Optional in-memory semantic route implemented by later model adapters."""

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        ...
