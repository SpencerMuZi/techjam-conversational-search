from __future__ import annotations

import re


class IntentRouter:
    """Lightweight, deterministic intent routing with override detection."""

    _override = re.compile(r"\b(actually|instead|ignore (?:my )?(?:earlier|previous))\b", re.I)
    _buying = re.compile(
        r"\b(key requirement|must have|what i need|need is|exactly|under \$?\d+|budget)\b",
        re.I,
    )
    _browsing = re.compile(r"\b(still exploring|browsing|open to|some ideas|not sure)\b", re.I)

    def classify(self, message: str, previous_route: str) -> tuple[str | None, bool]:
        if self._override.search(message):
            return "buying", True
        if self._buying.search(message):
            return "buying", False
        if self._browsing.search(message):
            return "browsing", False
        return previous_route or None, False
