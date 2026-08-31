from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from .clarification import ClarificationPolicy
from .config import AgentConfig
from .context import ContextBuilder
from .learned_reranker import default_reranker
from .retrieval import HybridRetriever
from .slots import SlotExtractor
from .state import SessionStore


class ShoppingCopilotAgent:
    """Official API implementation backed by adaptive dialogue orchestration."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.sessions = SessionStore()
        self.extractor = SlotExtractor()
        self.context_builder = ContextBuilder()
        self.retriever = HybridRetriever(catalog_path, self.config)
        self.retriever.reranker = default_reranker(self.config.learned_reranker)
        pool_depth = getattr(self.retriever.reranker, "meta", {}).get("pool_depth")
        if pool_depth:
            self.retriever.config = replace(
                self.config,
                retrieval_depth=max(self.config.retrieval_depth, int(pool_depth)),
                rerank_depth=max(self.config.rerank_depth, int(pool_depth)),
                precise_seed=max(self.config.precise_seed, int(pool_depth)),
            )
        self.clarification = ClarificationPolicy(self.config)
        self._early_config = replace(
            self.retriever.config,
            retrieval_depth=max(self.retriever.config.retrieval_depth, self.config.early_pool_depth),
            rerank_depth=max(self.retriever.config.rerank_depth, self.config.early_pool_depth),
            precise_seed=max(self.retriever.config.precise_seed, self.config.early_pool_depth),
            early_pool_enabled=False,
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        parsed = self.extractor.parse(user_message, state.route, state.last_ask_attribute)
        state.apply(parsed, user_message, turn)

        context = self.context_builder.build(state, user_message)
        result = self.retriever.retrieve(context, max(1, min(int(top_k), 10)))
        if self._should_probe_early_pool(turn, parsed.is_override):
            expanded = self.retriever.retrieve(context, 2, self._early_config)
            result = self._promote_confident_early_top(
                result,
                expanded,
                max(1, min(int(top_k), 10)),
                self.config.early_pool_margin,
            )
        state.candidate_count = result.candidate_count

        ask_attribute = self.clarification.choose(state, turn)
        message = self.clarification.message_for(ask_attribute, len(result.recommendations))
        state.record_response(message, ask_attribute)

        recommendations = result.recommendations
        if self._should_defer_recommendations(
            state, turn, ask_attribute, result.recommendations
        ):
            recommendations = []

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 8)}
                for parent_asin, score in recommendations
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _should_probe_early_pool(self, turn: int, is_override: bool) -> bool:
        reranker = self.retriever.reranker
        return bool(
            self.config.early_pool_enabled
            and reranker is not None
            and getattr(reranker, "meta", {}).get("model") == "lightgbm_lambdarank"
            and getattr(reranker, "meta", {}).get("variant") != "wide"
            and (turn == 1 or is_override)
        )

    @staticmethod
    def _promote_confident_early_top(
        base,
        expanded,
        top_k: int,
        margin_threshold: float,
    ):
        """Promote only a decisive wide-pool winner; preserve base ordering.

        Returning the whole wide-pool Top-10 lowers MTTC but can lock a relevant
        product into a weak reciprocal rank. A single-item promotion captures
        confident early wins while every non-promoted item keeps the precise
        model's original order.
        """
        from .retrieval import RetrievalResult

        if len(expanded.recommendations) < 2:
            return base
        margin = expanded.recommendations[0][1] - expanded.recommendations[1][1]
        if margin < margin_threshold:
            return base
        winner = expanded.recommendations[0]
        merged = [winner]
        merged.extend(item for item in base.recommendations if item[0] != winner[0])
        return RetrievalResult(
            recommendations=merged[:top_k],
            candidate_count=max(base.candidate_count, expanded.candidate_count),
        )

    @staticmethod
    def _should_defer_recommendations(
        state,
        turn: int,
        ask_attribute: str | None,
        recommendations,
    ) -> bool:
        """Avoid locking in a weak rank before the first useful preference.

        A first request typically leaves hundreds of plausible products, even
        when it carries one broad constraint. Returning that list immediately is
        poor commerce UX and also prevents the next clarification from improving
        the decision. We still run retrieval so the policy can inspect candidate
        volume, but hold the list for one turn while asking for another concrete
        preference.
        """
        mode = os.environ.get("SHOPPING_COPILOT_DEFERRAL", "none").lower()
        if mode == "none":
            return False
        if mode == "all":
            return turn == 1 and ask_attribute is not None
        if mode == "adaptive":
            if turn != 1 or ask_attribute is None or len(recommendations) < 2:
                return False
            top_score = float(recommendations[0][1])
            margin = top_score - float(recommendations[1][1])
            if state.route == "buying":
                return top_score >= 6.85
            return (
                not state.hard_slots
                and not state.soft_slots
                and top_score <= 8.62
                and margin <= 3.01
            )
        return (
            turn == 1
            and state.route == "browsing"
            and not state.hard_slots
            and not state.soft_slots
            and ask_attribute is not None
        )
