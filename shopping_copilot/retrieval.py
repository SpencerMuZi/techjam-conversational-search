from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import AgentConfig
from .features import RetrievalSignals, candidate_features, constraint_matches
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
        self.doc_terms: dict[str, frozenset[str]] = {}
        self.idf: dict[str, float] = {}
        self.default_idf: float = 0.0
        self._build()

    def term_weight(self, value: str) -> float:
        """Mean inverse document frequency of a constraint fragment's terms.

        Rare, discriminative phrases (``dual gore panels``) outweigh boilerplate
        that half the catalog shares (``imported``, ``leather``).
        """
        terms = _terms(value)
        if not terms:
            return 0.4
        weight = sum(self.idf.get(term, self.default_idf) for term in terms) / len(terms)
        return max(0.35, min(weight / 4.0, 2.5))

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

        document_frequency: dict[str, int] = {}
        for parent_asin, document in self.documents.items():
            terms = frozenset(_terms(document.searchable_text, limit=4000))
            self.doc_terms[parent_asin] = terms
            for term in terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        total = max(1, len(self.documents))
        self.idf = {
            term: math.log(total / (1 + count))
            for term, count in document_frequency.items()
        }
        self.default_idf = math.log(total / 1.0)

        self._popular = sorted(
            self.documents,
            key=lambda asin: (
                self.documents[asin].rating_number,
                self.documents[asin].average_rating,
            ),
            reverse=True,
        )

    def search_fts(self, query: str, limit: int) -> list[str]:
        return [asin for asin, _ in self.search_fts_scored(query, limit)]

    def search_fts_scored(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Ranked ``(parent_asin, relevance)`` pairs; higher relevance is better.

        SQLite ``bm25()`` returns smaller-is-better negative numbers, so the sign
        is flipped here to give a monotone "higher = closer match" score that the
        reranker features can consume directly.
        """
        terms = _terms(query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 7.0, 5.0, 3.0, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(asin), -float(score)) for asin, score in rows]

    def search_fts_all(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Require every informative term from one constraint fragment.

        The broad route intentionally uses OR for recall. This companion route
        recovers products containing an entire disclosed feature phrase, which
        would otherwise be buried under thousands of partial matches for common
        words such as ``cotton`` or ``grey``.
        """
        terms = _terms(query)[:16]
        if len(terms) < 2:
            return []
        expression = " AND ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 7.0, 5.0, 3.0, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(asin), -float(score)) for asin, score in rows]

    def popular(self, limit: int) -> list[str]:
        return self._popular[:limit]


class HybridRetriever:
    """Fuses keyword, category, constraint, and optional semantic routes."""

    def __init__(
        self,
        catalog_path: str | Path,
        config: AgentConfig,
        semantic: SemanticRetriever | None = None,
        reranker=None,
    ) -> None:
        self.config = config
        self.index = CatalogIndex(catalog_path)
        self.semantic = semantic or NullSemanticRetriever()
        # Optional learned reranker: ``score(feature_rows) -> list[float]`` where a
        # higher value ranks earlier. ``None`` keeps the manual structured score.
        self.reranker = reranker
        # When True, ``retrieve`` stashes the exact (asin, feature vector) pool it
        # scored, so an offline experiment can build training data from it.
        self.capture = False
        self.last_candidates: list[tuple[str, list[float]]] = []
        self.last_context = None
        self.last_signals = None

    def retrieve(self, context: SearchContext, top_k: int) -> RetrievalResult:
        depth = self.config.retrieval_depth
        weights = self.config.weights_for(context.route)
        constraint_text = " ".join((*context.hard_values, *context.soft_values))
        keyword_text = " ".join(
            (context.current_message, context.category, constraint_text, " ".join(context.profile_tags))
        )
        semantic_query = " ".join((context.category, constraint_text, " ".join(context.profile_tags))).strip()

        keyword_rows = self.index.search_fts_scored(keyword_text, depth)
        category_rows = self.index.search_fts_scored(context.category, depth)
        constraint_rows = self.index.search_fts_scored(constraint_text, depth)
        # Put full-fragment matches ahead of the broad OR route. Process the most
        # specific fragments first so a long product feature outranks a generic
        # material or colour term.
        fragments = sorted(
            (*context.hard_values, *context.soft_values),
            key=lambda value: (
                len(_terms(value)),
                self.index.term_weight(value),
            ),
            reverse=True,
        )
        conjunctive_rows: list[tuple[str, float]] = []
        conjunctive_seen: set[str] = set()
        per_fragment = max(10, min(50, depth // 4))
        for fragment in fragments:
            for asin, score in self.index.search_fts_all(fragment, per_fragment):
                if asin not in conjunctive_seen:
                    conjunctive_seen.add(asin)
                    conjunctive_rows.append((asin, score))
        if conjunctive_rows:
            constraint_rows = conjunctive_rows + [
                row for row in constraint_rows if row[0] not in conjunctive_seen
            ]
        semantic_rows = self.semantic.search(semantic_query, depth) if semantic_query else []

        keyword_ranked = [asin for asin, _ in keyword_rows]
        constraint_ranked = [asin for asin, _ in constraint_rows]

        routes: list[tuple[float, list[str]]] = [
            (weights.keyword, keyword_ranked),
            (weights.category, [asin for asin, _ in category_rows]),
            (weights.constraints, constraint_ranked),
            (weights.semantic, [asin for asin, _ in semantic_rows]),
        ]

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

        # Best position across the two *precise* routes. A target that BM25 put
        # near the top of keyword or constraint search should not be reranked
        # out of the Top-10 by broad category / popularity offsets.
        precise_rank: dict[str, int] = {}
        for ranked in (keyword_ranked, constraint_ranked):
            for rank, parent_asin in enumerate(ranked, start=1):
                if rank < precise_rank.get(parent_asin, 10**9):
                    precise_rank[parent_asin] = rank

        candidate_count = len(fused)
        initial = sorted(fused, key=fused.get, reverse=True)[: self.config.rerank_depth]
        # Guarantee the head of each precise route reaches the reranker even if a
        # thin fused score would otherwise drop it below the rerank cutoff.
        seen = set(initial)
        for ranked in (keyword_ranked, constraint_ranked):
            for parent_asin in ranked[: self.config.precise_seed]:
                if parent_asin not in seen and parent_asin in self.index.documents:
                    seen.add(parent_asin)
                    initial.append(parent_asin)

        if self.reranker is not None or self.capture:
            signals = RetrievalSignals.from_routes(
                keyword_rows, category_rows, constraint_rows, fused, getattr(context, "turn", 1)
            )
            feature_rows = [
                candidate_features(self.index, self.index.documents[asin], context, signals)
                for asin in initial
            ]
            if self.capture:
                self.last_candidates = list(zip(initial, feature_rows))
                self.last_context = context
                self.last_signals = signals
            if self.reranker is not None:
                model_scores = self.reranker.score(feature_rows)
                scored = list(zip(initial, (float(s) for s in model_scores)))
                scored.sort(key=lambda item: item[1], reverse=True)
                return RetrievalResult(recommendations=scored[:top_k], candidate_count=candidate_count)

        scored = [
            (
                asin,
                fused[asin]
                + self._structured_score(self.index.documents[asin], context, precise_rank.get(asin)),
            )
            for asin in initial
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return RetrievalResult(recommendations=scored[:top_k], candidate_count=candidate_count)

    def _structured_score(
        self,
        product: ProductDocument,
        context: SearchContext,
        precise_rank: int | None = None,
    ) -> float:
        score = 0.0

        # Retrieval prior: decaying reward for a strong BM25 position, so a target
        # that keyword/constraint search ranked at the very top is nudged toward
        # rank 1 instead of being flattened into a pack of category peers.
        if precise_rank is not None:
            score += 1.3 / (1.0 + math.log1p(precise_rank))

        category_terms = set(_terms(context.category))
        product_category_terms = set(_terms(product.categories, limit=100))
        if category_terms:
            score += 0.8 * len(category_terms & product_category_terms) / len(category_terms)

        # Constraint matching: base framework's flat reward plus a small idf bonus
        # that breaks ties in favour of the item matching the rarer phrase.
        for attribute, values in context.hard_slots.items():
            for value in values:
                if self._matches(product, attribute, value):
                    score += 1.6 + 0.35 * self.index.term_weight(value)
                else:
                    score -= 0.7
        for attribute, values in context.soft_slots.items():
            for value in values:
                if self._matches(product, attribute, value):
                    score += 0.4 + 0.15 * self.index.term_weight(value)

        profile_hits = sum(
            1 for tag in context.profile_tags if tag.lower() in product.searchable_text
        )
        score += min(profile_hits, 3) * 0.025
        score += min(math.log1p(product.rating_number) / 20.0, 0.08)
        score += max(0.0, min(product.average_rating / 5.0, 1.0)) * 0.025
        return score

    def _matches(self, product: ProductDocument, attribute: str, value: str) -> bool:
        return constraint_matches(self.index, product, attribute, value)
