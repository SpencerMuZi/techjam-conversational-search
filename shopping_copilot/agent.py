from __future__ import annotations

import os
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
        self.clarification = ClarificationPolicy(self.config)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        parsed = self.extractor.parse(user_message, state.route, state.last_ask_attribute)
        state.apply(parsed, user_message, turn)

        context = self.context_builder.build(state, user_message)
        result = self.retriever.retrieve(context, max(1, min(int(top_k), 10)))
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
