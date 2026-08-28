from __future__ import annotations

import re

from .intent import IntentRouter
from .models import ParsedTurn


MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
ATTRIBUTE_WORDS = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if "color" in lowered or any(re.search(rf"\b{re.escape(color)}\b", lowered) for color in COLORS):
        return "color"
    if re.search(r"\b(size|sizing|width|wide|narrow|small|medium|large|xl|xxl)\b", lowered):
        return "size"
    if re.search(r"\b(department|style|fit|sleeve|neck|casual|formal)\b", lowered):
        return "style"
    if re.search(r"\b(hiking|running|gym|winter|outdoor|work|walking)\b", lowered):
        return "use_case"
    return "feature"


class SlotExtractor:
    _category = re.compile(r"\blooking for\s+(.+?)(?:,|\.\s|$)", re.I)
    _constraint = re.compile(r"(?:key requirement is|what matters is|what i need is):?\s*(.+)$", re.I)
    _no_preference = re.compile(
        r"(?:no|don't have (?:an? )?)(?:additional )?preference for\s+([a-z_]+)",
        re.I,
    )

    def __init__(self, router: IntentRouter | None = None) -> None:
        self.router = router or IntentRouter()

    def parse(self, message: str, previous_route: str, last_ask_attribute: str | None) -> ParsedTurn:
        route_hint, is_override = self.router.classify(message, previous_route)
        category_match = self._category.search(message)
        category = category_match.group(1).strip() if category_match else None

        no_preference = None
        no_preference_match = self._no_preference.search(message)
        if no_preference_match:
            candidate = no_preference_match.group(1).lower()
            no_preference = candidate if candidate in ATTRIBUTE_WORDS else last_ask_attribute
        elif "don't have a preference" in message.lower():
            no_preference = last_ask_attribute

        slots: dict[str, list[str]] = {}
        fragments = self._constraint_fragments(message, category_match)
        for fragment in fragments:
            attribute = classify_constraint(fragment)
            slots.setdefault(attribute, []).append(fragment)

        return ParsedTurn(
            route_hint=route_hint,
            category=category,
            slots=slots,
            no_preference=no_preference,
            is_override=is_override,
        )

    def _constraint_fragments(self, message: str, category_match: re.Match[str] | None) -> list[str]:
        match = self._constraint.search(message)
        if match:
            raw = match.group(1)
        elif message.lower().startswith("for that, what matters is:"):
            raw = message.split(":", 1)[1]
        elif category_match:
            raw = message[category_match.end():].strip(" .,;")
            if raw.lower().startswith("but i'm still exploring"):
                return []
        else:
            return []

        fragments = []
        for value in raw.split(";"):
            cleaned = re.sub(r"\s+", " ", value).strip(" .,;\t\n")
            if cleaned and "don't have" not in cleaned.lower() and "not quite right" not in cleaned.lower():
                fragments.append(cleaned[:240])
        return fragments
