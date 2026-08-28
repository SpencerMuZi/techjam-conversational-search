from __future__ import annotations

from .models import SearchContext
from .state import ConversationState


class ContextBuilder:
    """Distills dialogue history into a compact, route-specific context."""

    def build(self, state: ConversationState, current_message: str) -> SearchContext:
        profile_tags = tuple(str(tag) for tag in state.user_profile.get("preference_tags", []) if tag)
        return SearchContext(
            route=state.route,
            category=state.category,
            hard_slots={key: tuple(values) for key, values in state.hard_slots.items()},
            soft_slots={key: tuple(values) for key, values in state.soft_slots.items()},
            profile_tags=profile_tags,
            current_message=current_message,
        )
