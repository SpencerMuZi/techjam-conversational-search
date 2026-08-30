from __future__ import annotations

from .config import AgentConfig
from .state import ConversationState


QUESTION_TEXT = {
    "other": "What other requirements or preferences should I prioritize?",
    "material": "Do you have a preferred material?",
    "feature": "Which feature matters most to you?",
    "color": "Do you have a color preference?",
    "style": "What style or fit would you prefer?",
    "size": "Are there any sizing or width requirements?",
    "use_case": "What will you mainly use it for?",
    "budget": "What budget range should I stay within?",
}


class ClarificationPolicy:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def choose(self, state: ConversationState, turn: int) -> str | None:
        if turn >= 10 or (state.candidate_count and state.candidate_count <= self.config.clarification_candidate_cutoff):
            return None

        known = set(state.hard_slots) | set(state.soft_slots)
        unavailable = set(state.asked_attributes) | state.no_preference_attributes
        order = self._priority_order(state)
        for attribute in order:
            if attribute not in known and attribute not in unavailable:
                return attribute
        for attribute in order:
            if attribute not in unavailable:
                return attribute
        return None

    def message_for(self, attribute: str | None, recommendation_count: int) -> str:
        if attribute:
            prefix = "I found a few possible matches." if recommendation_count else "I need one more detail."
            return f"{prefix} {QUESTION_TEXT[attribute]}"
        return "Here are the strongest matches based on what you have told me."

    def _priority_order(self, state: ConversationState) -> tuple[str, ...]:
        profile_tags = {str(tag).lower() for tag in state.user_profile.get("preference_tags", [])}
        preferred: list[str] = []
        if "material" in profile_tags:
            preferred.append("material")
        if "fit" in profile_tags:
            preferred.append("style")
        if "performance" in profile_tags or "weather" in profile_tags:
            preferred.append("use_case")

        base = (
            ("other", "feature", "material", "color", "style", "size", "use_case", "budget")
            if state.route == "buying" and "material" in state.hard_slots
            else ("other", "material", "feature", "color", "style", "size", "use_case", "budget")
        )
        return tuple(dict.fromkeys([*preferred, *base]))
