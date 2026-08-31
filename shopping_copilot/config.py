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
    rrf_k: int = 60
    clarification_candidate_cutoff: int = 20
    max_query_terms: int = 48
    # Load shopping_copilot/reranker_lr.json (stdlib-only linear model) as the
    # reranker when present. Set False, or SHOPPING_COPILOT_RERANKER=manual, to
    # fall back to the hand-tuned structured score.
    learned_reranker: bool = True
    # Optional local cross-encoder re-scoring the linear reranker's shortlist.
    # Off by default: it needs sentence-transformers + model weights. Enable via
    # AgentConfig(cross_encoder=True) or SHOPPING_COPILOT_CROSS_ENCODER=1.
    cross_encoder: bool = False
    cross_encoder_depth: int = 20
    cross_encoder_weight: float = 0.7
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
