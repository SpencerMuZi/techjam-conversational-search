from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import AgentConfig
from .models import ProductDocument, SearchContext, SemanticRetriever
from .semantic import NullSemanticRetriever


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "what", "matters", "key", "requirement", "those",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(value: str, limit: int = 48) -> list[str]:
    return list(
        dict.fromkeys(
            token.lower()
            for token in TOKEN_RE.findall(value)
            if len(token) > 1 and token.lower() not in STOPWORDS
        )
    )[:limit]


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RetrievalResult:
    recommendations: list[tuple[str, float]]
    candidate_count: int


class CatalogIndex:
    """In-memory catalog plus FTS5 indexes for heterogeneous retrieval routes."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.documents: dict[str, ProductDocument] = {}
        self._popular: list[str] = []
        self._build()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                searchable = " ".join((title, categories, features, details, store, description)).lower()
                self.documents[parent_asin] = ProductDocument(
                    parent_asin=parent_asin,
                    title=title,
                    categories=categories,
                    features=features,
                    details=details,
                    store=store,
                    description=description,
                    price=_as_float(product.get("price"), default=-1.0) if product.get("price") not in (None, "") else None,
                    average_rating=_as_float(product.get("average_rating")),
                    rating_number=int(_as_float(product.get("rating_number"))),
                    searchable_text=searchable,
                )
                batch.append((parent_asin, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self._popular = sorted(
            self.documents,
            key=lambda asin: (
                self.documents[asin].rating_number,
                self.documents[asin].average_rating,
            ),
            reverse=True,
        )

    def search_fts(self, query: str, limit: int) -> list[str]:
        terms = _terms(query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 7.0, 5.0, 3.0, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def popular(self, limit: int) -> list[str]:
        return self._popular[:limit]


class HybridRetriever:
    """Fuses keyword, category, constraint, and optional semantic routes."""

    def __init__(
        self,
        catalog_path: str | Path,
        config: AgentConfig,
        semantic: SemanticRetriever | None = None,
    ) -> None:
        self.config = config
        self.index = CatalogIndex(catalog_path)
        self.semantic = semantic or NullSemanticRetriever()
        self._term_cache: dict[str, frozenset[str]] = {}

    def retrieve(self, context: SearchContext, top_k: int) -> RetrievalResult:
        depth = self.config.retrieval_depth
        weights = self.config.weights_for(context.route)
        constraint_text = " ".join((*context.hard_values, *context.soft_values))
        keyword_text = " ".join(
            (context.current_message, context.category, constraint_text, " ".join(context.profile_tags))
        )
        semantic_query = " ".join((context.category, constraint_text, " ".join(context.profile_tags))).strip()

        routes: list[tuple[float, list[str]]] = [
            (weights.keyword, self.index.search_fts(keyword_text, depth)),
            (weights.category, self.index.search_fts(context.category, depth)),
            (weights.constraints, self.index.search_fts(constraint_text, depth)),
        ]
        semantic_rows = self.semantic.search(semantic_query, depth) if semantic_query else []
        routes.append((weights.semantic, [asin for asin, _ in semantic_rows]))

        fused: dict[str, float] = {}
        for weight, ranked in routes:
            if weight <= 0:
                continue
            for rank, parent_asin in enumerate(ranked, start=1):
                if parent_asin in self.index.documents:
                    fused[parent_asin] = fused.get(parent_asin, 0.0) + weight / (self.config.rrf_k + rank)

        if not fused:
            fused = {
                asin: 1.0 / (self.config.rrf_k + rank)
                for rank, asin in enumerate(self.index.popular(depth), start=1)
            }

        candidate_count = len(fused)
        initial = sorted(fused, key=fused.get, reverse=True)[: self.config.rerank_depth]
        scored = [
            (asin, fused[asin] + self._structured_score(self.index.documents[asin], context))
            for asin in initial
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return RetrievalResult(recommendations=scored[:top_k], candidate_count=candidate_count)

    def _structured_score(self, product: ProductDocument, context: SearchContext) -> float:
        score = 0.0
        category_terms = set(_terms(context.category))
        product_category_terms = set(_terms(product.categories, limit=100))
        if category_terms:
            score += 0.8 * len(category_terms & product_category_terms) / len(category_terms)

        for attribute, values in context.hard_slots.items():
            for value in values:
                score += 1.6 if self._matches(product, attribute, value) else -0.7
        for attribute, values in context.soft_slots.items():
            for value in values:
                score += 0.4 if self._matches(product, attribute, value) else 0.0

        profile_hits = sum(
            1 for tag in context.profile_tags if tag.lower() in product.searchable_text
        )
        score += min(profile_hits, 3) * 0.025
        score += min(math.log1p(product.rating_number) / 20.0, 0.08)
        score += max(0.0, min(product.average_rating / 5.0, 1.0)) * 0.025
        return score

    def _matches(self, product: ProductDocument, attribute: str, value: str) -> bool:
        if attribute == "budget" and product.price is not None:
            amount_match = re.search(r"(?:\$|around\s*\$?|under\s*\$?)\s*(\d+(?:\.\d+)?)", value, re.I)
            if amount_match:
                amount = float(amount_match.group(1))
                return product.price <= amount if "under" in value.lower() else abs(product.price - amount) <= max(10.0, amount * 0.3)

        lowered = value.lower().strip()
        if lowered and lowered in product.searchable_text:
            return True
        terms = set(_terms(value))
        if not terms:
            return False
        product_terms = self._term_cache.get(product.parent_asin)
        if product_terms is None:
            product_terms = frozenset(_terms(product.searchable_text, limit=10000))
            self._term_cache[product.parent_asin] = product_terms
        required = max(1, math.ceil(len(terms) * 0.6))
        return len(terms & product_terms) >= required
