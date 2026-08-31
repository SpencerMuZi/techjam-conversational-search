from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RouteWeights:
    """Weights used to fuse retrieval routes for one intent."""

    keyword: float
    category: float
    constraints: float
    semantic: float


@dataclass(frozen=True)
class AgentConfig:
    """Central configuration for retrieval and dialogue orchestration."""

    retrieval_depth: int = 250
    rerank_depth: int = 120
    precise_seed: int = 100
    # A wider pool is consulted only on the first turn and immediately after an
    # explicit intent override. Its top item may be promoted when the learned
    # ranker has a clear lead, improving time-to-conversion without replacing
    # the precise 120-candidate ranking used on every turn.
    early_pool_enabled: bool = True
    early_pool_depth: int = 300
    early_pool_margin: float = 0.60
    rrf_k: int = 60
    clarification_candidate_cutoff: int = 20
    max_query_terms: int = 48
    # Load shopping_copilot/reranker_lr.json (stdlib-only linear model) as the
    # reranker when present. Set False, or SHOPPING_COPILOT_RERANKER=manual, to
    # fall back to the hand-tuned structured score.
    learned_reranker: bool = True
    buying_weights: RouteWeights = field(
        default_factory=lambda: RouteWeights(
            keyword=1.0,
            category=1.2,
            constraints=2.0,
            semantic=0.5,
        )
    )
    browsing_weights: RouteWeights = field(
        default_factory=lambda: RouteWeights(
            keyword=0.8,
            category=1.2,
            constraints=1.2,
            semantic=1.2,
        )
    )

    def weights_for(self, route: str) -> RouteWeights:
        return self.buying_weights if route == "buying" else self.browsing_weights
