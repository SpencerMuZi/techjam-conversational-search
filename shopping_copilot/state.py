from __future__ import annotations

from dataclasses import dataclass, field

from .models import ParsedTurn


@dataclass
class ConversationState:
    session_id: str
    user_profile: dict
    route: str = "browsing"
    category: str = ""
    hard_slots: dict[str, list[str]] = field(default_factory=dict)
    soft_slots: dict[str, list[str]] = field(default_factory=dict)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    history: list[tuple[str, str]] = field(default_factory=list)
    last_ask_attribute: str | None = None
    candidate_count: int = 0
    turn: int = 1

    def apply(self, parsed: ParsedTurn, user_message: str, turn: int) -> None:
        self.turn = turn
        if parsed.is_override:
            # The customer explicitly invalidated the earlier intent. Category and
            # profile are stable; intent-specific slots must be reconstructed.
            self.hard_slots.clear()
            self.soft_slots.clear()
            self.no_preference_attributes.clear()
            self.route = "buying"
        elif parsed.route_hint:
            self.route = parsed.route_hint

        if parsed.category:
            self.category = parsed.category

        if parsed.no_preference:
            self.no_preference_attributes.add(parsed.no_preference)

        destination = self.hard_slots if self.route == "buying" or parsed.is_override else self.soft_slots
        for attribute, values in parsed.slots.items():
            if attribute in self.no_preference_attributes:
                continue
            bucket = destination.setdefault(attribute, [])
            for value in values:
                if value and value not in bucket:
                    bucket.append(value)

        self.history.append(("user", user_message))

    def record_response(self, message: str, ask_attribute: str | None) -> None:
        self.history.append(("assistant", message))
        self.last_ask_attribute = ask_attribute
        if ask_attribute and ask_attribute not in self.asked_attributes:
            self.asked_attributes.append(ask_attribute)


class SessionStore:
    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def reset(self, session_id: str, user_profile: dict) -> ConversationState:
        state = ConversationState(session_id=session_id, user_profile=dict(user_profile))
        self._states[session_id] = state
        return state

    def get(self, session_id: str) -> ConversationState:
        try:
            return self._states[session_id]
        except KeyError as exc:
            raise RuntimeError("reset must be called before respond") from exc
