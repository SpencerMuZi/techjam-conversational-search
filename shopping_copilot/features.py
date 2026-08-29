"""Shared candidate feature extraction for every reranking strategy.

The manual score, logistic regression, random forest, and LambdaRank models all
consume the *same* vector produced here, so the model comparison isolates the
ranking function and never the information it is given.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .models import ProductDocument, SearchContext

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "what", "matters", "key", "requirement", "those",
}

_BUDGET_RE = re.compile(r"(?:\$|around\s*\$?|under\s*\$?)\s*(\d+(?:\.\d+)?)", re.I)


def terms(value: str, limit: int = 48) -> list[str]:
    return list(
        dict.fromkeys(
            token.lower()
            for token in TOKEN_RE.findall(value)
            if len(token) > 1 and token.lower() not in _STOPWORDS
        )
    )[:limit]


def constraint_matches(index, product: ProductDocument, attribute: str, value: str) -> bool:
    """True when ``product`` satisfies a single disclosed constraint fragment."""
    if attribute == "budget" and product.price is not None:
        amount_match = _BUDGET_RE.search(value)
        if amount_match:
            amount = float(amount_match.group(1))
            if "under" in value.lower():
                return product.price <= amount
            return abs(product.price - amount) <= max(10.0, amount * 0.3)

    lowered = value.lower().strip()
    if lowered and lowered in product.searchable_text:
        return True
    wanted = set(terms(value))
    if not wanted:
        return False
    product_terms = index.doc_terms.get(product.parent_asin)
    if product_terms is None:
        product_terms = frozenset(terms(product.searchable_text, limit=10000))
    required = max(1, math.ceil(len(wanted) * 0.6))
    return len(wanted & product_terms) >= required


@dataclass
class RetrievalSignals:
    """Per-candidate retrieval evidence gathered before the final rerank."""

    keyword_rank: dict[str, int] = field(default_factory=dict)
    keyword_score: dict[str, float] = field(default_factory=dict)
    category_rank: dict[str, int] = field(default_factory=dict)
    category_score: dict[str, float] = field(default_factory=dict)
    constraint_rank: dict[str, int] = field(default_factory=dict)
    constraint_score: dict[str, float] = field(default_factory=dict)
    fused: dict[str, float] = field(default_factory=dict)
    turn: int = 1

    @classmethod
    def from_routes(cls, keyword, category, constraint, fused, turn):
        def rank_map(rows):
            return {asin: i for i, (asin, _) in enumerate(rows, start=1)}

        def score_map(rows):
            return {asin: s for asin, s in rows}

        return cls(
            keyword_rank=rank_map(keyword), keyword_score=score_map(keyword),
            category_rank=rank_map(category), category_score=score_map(category),
            constraint_rank=rank_map(constraint), constraint_score=score_map(constraint),
            fused=dict(fused), turn=turn,
        )

    def precise_best_rank(self, asin: str) -> int | None:
        ranks = [r for r in (self.keyword_rank.get(asin), self.constraint_rank.get(asin)) if r]
        return min(ranks) if ranks else None


# Canonical feature order. Everything downstream indexes by this list.
FEATURE_NAMES: list[str] = [
    # constraint satisfaction
    "hard_total", "hard_hits", "hard_hit_rate", "hard_miss_count",
    "hard_idf_sum", "hard_idf_max",
    "soft_total", "soft_hits", "soft_hit_rate", "soft_idf_sum",
    "full_coverage", "coverage_with_soft", "exact_phrase_hits",
    # category
    "category_overlap_frac", "category_overlap_count",
    # retrieval evidence
    "kw_present", "kw_recip", "kw_score",
    "cat_present", "cat_recip", "cat_score",
    "constraint_present", "constraint_recip", "constraint_score",
    "rrf_fused_score", "precise_best_recip", "precise_prior", "route_presence_count",
    # profile / quality
    "profile_hits", "profile_hits_capped", "avg_rating", "log_rating_number",
    "has_price", "budget_match",
    # dialogue context
    "turn", "route_is_buying",
]


def candidate_features(
    index,
    product: ProductDocument,
    context: SearchContext,
    signals: RetrievalSignals,
) -> list[float]:
    asin = product.parent_asin

    hard_total = hard_hits = hard_miss = 0
    hard_idf_sum = hard_idf_max = 0.0
    exact_phrase_hits = 0
    for attribute, values in context.hard_slots.items():
        for value in values:
            hard_total += 1
            weight = index.term_weight(value)
            if value.lower().strip() and value.lower().strip() in product.searchable_text:
                exact_phrase_hits += 1
            if constraint_matches(index, product, attribute, value):
                hard_hits += 1
                hard_idf_sum += weight
                hard_idf_max = max(hard_idf_max, weight)
            else:
                hard_miss += 1

    soft_total = soft_hits = 0
    soft_idf_sum = 0.0
    for attribute, values in context.soft_slots.items():
        for value in values:
            soft_total += 1
            if constraint_matches(index, product, attribute, value):
                soft_hits += 1
                soft_idf_sum += index.term_weight(value)

    full_coverage = 1.0 if hard_total and hard_hits == hard_total else 0.0
    coverage_with_soft = 1.0 if full_coverage and soft_hits else 0.0

    category_terms = set(terms(context.category))
    product_category_terms = set(terms(product.categories, limit=100))
    overlap = len(category_terms & product_category_terms)
    category_overlap_frac = overlap / len(category_terms) if category_terms else 0.0

    kw_rank = signals.keyword_rank.get(asin)
    cat_rank = signals.category_rank.get(asin)
    con_rank = signals.constraint_rank.get(asin)
    precise_rank = signals.precise_best_rank(asin)
    route_presence = sum(x is not None for x in (kw_rank, cat_rank, con_rank))

    profile_hits = sum(
        1 for tag in context.profile_tags if tag.lower() in product.searchable_text
    )

    has_price = 1.0 if product.price is not None and product.price >= 0 else 0.0
    budget_match = 0.0
    for values in context.hard_slots.values():
        for value in values:
            if _BUDGET_RE.search(value) and constraint_matches(index, product, "budget", value):
                budget_match = 1.0

    values = {
        "hard_total": float(hard_total),
        "hard_hits": float(hard_hits),
        "hard_hit_rate": hard_hits / hard_total if hard_total else 0.0,
        "hard_miss_count": float(hard_miss),
        "hard_idf_sum": hard_idf_sum,
        "hard_idf_max": hard_idf_max,
        "soft_total": float(soft_total),
        "soft_hits": float(soft_hits),
        "soft_hit_rate": soft_hits / soft_total if soft_total else 0.0,
        "soft_idf_sum": soft_idf_sum,
        "full_coverage": full_coverage,
        "coverage_with_soft": coverage_with_soft,
        "exact_phrase_hits": float(exact_phrase_hits),
        "category_overlap_frac": category_overlap_frac,
        "category_overlap_count": float(overlap),
        "kw_present": 1.0 if kw_rank else 0.0,
        "kw_recip": 1.0 / kw_rank if kw_rank else 0.0,
        "kw_score": signals.keyword_score.get(asin, 0.0),
        "cat_present": 1.0 if cat_rank else 0.0,
        "cat_recip": 1.0 / cat_rank if cat_rank else 0.0,
        "cat_score": signals.category_score.get(asin, 0.0),
        "constraint_present": 1.0 if con_rank else 0.0,
        "constraint_recip": 1.0 / con_rank if con_rank else 0.0,
        "constraint_score": signals.constraint_score.get(asin, 0.0),
        "rrf_fused_score": signals.fused.get(asin, 0.0),
        "precise_best_recip": 1.0 / precise_rank if precise_rank else 0.0,
        "precise_prior": 1.0 / (1.0 + math.log1p(precise_rank)) if precise_rank else 0.0,
        "route_presence_count": float(route_presence),
        "profile_hits": float(profile_hits),
        "profile_hits_capped": float(min(profile_hits, 3)),
        "avg_rating": product.average_rating,
        "log_rating_number": math.log1p(product.rating_number),
        "has_price": has_price,
        "budget_match": budget_match,
        "turn": float(signals.turn),
        "route_is_buying": 1.0 if context.route == "buying" else 0.0,
    }
    return [values[name] for name in FEATURE_NAMES]
