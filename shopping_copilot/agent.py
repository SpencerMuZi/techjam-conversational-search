from __future__ import annotations

from pathlib import Path

from .clarification import ClarificationPolicy
from .config import AgentConfig
from .context import ContextBuilder
from .cross_encoder import load_cross_encoder
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
        self.retriever.cross_encoder = load_cross_encoder(
            self.config.cross_encoder,
            depth=self.config.cross_encoder_depth,
            weight=self.config.cross_encoder_weight,
        )
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

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 8)}
                for parent_asin, score in result.recommendations
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
