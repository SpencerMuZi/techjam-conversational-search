from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:budget\s+around|under|below|\$)\s*\$?(\d+(?:\.\d+)?)", re.I)
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "want",
    "with", "would", "you", "looking", "about", "additional", "actually", "again", "all", "any", "can",
    "could", "detail", "don", "earlier", "have", "ignore", "more", "need", "not", "now", "preference",
    "quite", "right", "still", "those", "what", "yet", "matters", "key", "requirement", "around",
}
PROFILE_TERMS = {
    "comfort": ("comfortable", "comfort", "soft", "lightweight", "cushion"),
    "durability": ("durable", "durability", "quality", "sturdy"),
    "fit": ("fit", "fitted", "size", "sizing", "wide", "narrow"),
    "material": MATERIALS,
    "performance": ("performance", "running", "sport", "athletic", "support"),
    "style": ("style", "fashion", "classic", "casual", "elegant"),
    "warmth": ("warm", "winter", "thermal", "insulated"),
    "weather": ("waterproof", "rain", "weather", "outdoor"),
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in STOPWORDS]


def _number(value: str) -> float | None:
    try:
        return float(value.replace("$", "").replace(",", "").strip())
    except (AttributeError, ValueError):
        return None


class Agent:
    """Stateful hybrid shopping ranker with slot-level intent replacement.

    All weights are fixed, hand-specified retrieval weights. The agent never
    consumes development labels, scenario names, or target ASINs, so the same
    logic can run on the organizer's unseen sessions.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict[str, object]] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, price, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]), _text(product.get("title")),
                    _text(product.get("categories")), _text(product.get("features")),
                    _text(product.get("details")), _text(product.get("store")),
                    _text(product.get("description")), _text(product.get("price")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "slots": {"category": []}, "profile": user_profile, "last_slot": None,
        }

    @staticmethod
    def _category_from_first_message(message: str) -> str:
        match = re.search(r"looking for\s+(.+?)(?:,\s*but\b|\.\s|$)", message, re.I)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _is_override(message: str) -> bool:
        return "ignore my earlier preference" in message.lower() or "actually, ignore" in message.lower()

    @staticmethod
    def _clauses(message: str) -> list[str]:
        patterns = (r"key requirement is:\s*(.+)$", r"what matters is:\s*(.+)$", r"what i need is:\s*(.+)$")
        for pattern in patterns:
            match = re.search(pattern, message, re.I)
            if match:
                return [part.strip(" .") for part in match.group(1).split(";") if part.strip(" .")]
        if "looking for" in message.lower() and "." in message:
            tail = message.split(".", 1)[1].strip(" .")
            return [tail] if tail and "still exploring" not in tail.lower() else []
        return []

    @staticmethod
    def _slot_for(value: str) -> str:
        lowered = value.lower()
        if PRICE_RE.search(lowered) or "budget" in lowered:
            return "budget"
        if any(word in lowered for word in MATERIALS):
            return "material"
        if any(word in lowered for word in COLORS) or "color" in lowered:
            return "color"
        if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
            return "size"
        if any(word in lowered for word in ("brand", "store", "manufacturer")):
            return "brand"
        if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
            return "use_case"
        if any(word in lowered for word in ("style", "fit", "sleeve", "neck", "dress", "casual")):
            return "style"
        return "feature"

    @staticmethod
    def _profile_affinity(product_text: str, profile: object) -> float:
        if not isinstance(profile, dict):
            return 0.0
        tags = profile.get("preference_tags", [])
        if not isinstance(tags, list):
            return 0.0
        matched = sum(any(term in product_text for term in PROFILE_TERMS.get(str(tag), ())) for tag in tags)
        return matched / max(1, len(tags))

    @staticmethod
    def _coverage(product_text: str, slot_values: list[str]) -> float:
        if not slot_values:
            return 0.0
        parts: list[float] = []
        for value in slot_values:
            phrase = value.lower()
            if phrase in product_text:
                parts.append(1.0)
            else:
                tokens = set(_terms(phrase))
                parts.append(sum(token in product_text for token in tokens) / max(1, len(tokens)))
        return sum(parts) / len(parts)

    @staticmethod
    def _budget_fit(price: str, slot_values: list[str]) -> float:
        target = next((PRICE_RE.search(value) for value in slot_values if PRICE_RE.search(value)), None)
        if target is None:
            return 0.0
        requested, actual = _number(target.group(1)), _number(price)
        if requested is None or actual is None:
            return 0.0
        return math.exp(-abs(actual - requested) / max(5.0, requested * 0.15))

    @staticmethod
    def _slot_text(slots: object) -> list[str]:
        if not isinstance(slots, dict):
            return []
        return [str(value) for values in slots.values() if isinstance(values, list) for value in values]

    def _recommend(self, state: dict[str, object], top_k: int) -> list[dict]:
        slots = state["slots"]
        values = self._slot_text(slots)
        terms = list(dict.fromkeys(_terms(" ".join(values))))[:60]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description, price "
            "FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 7.0, 5.0, 3.0, 2.0, 2.0, 1.5, 1.0) LIMIT 300",
            (expression,),
        ).fetchall()
        scored: list[tuple[float, str]] = []
        for rank, row in enumerate(rows):
            parent_asin, *fields = row
            product_text = " ".join(str(field or "") for field in fields[:-1]).lower()
            lexical = 1.0 - rank / max(1, len(rows))
            coverage = self._coverage(product_text, values)
            budget = self._budget_fit(str(fields[-1] or ""), values)
            profile = self._profile_affinity(product_text, state.get("profile"))
            score = 0.55 * lexical + 0.30 * coverage + 0.10 * budget + 0.05 * profile
            scored.append((score, str(parent_asin)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{"parent_asin": asin, "score": score} for score, asin in scored[:top_k]]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        slots = state["slots"]
        if not isinstance(slots, dict):
            raise RuntimeError("invalid session state")
        if turn == 1:
            category = self._category_from_first_message(user_message)
            if category:
                slots["category"] = [category]

        clauses = self._clauses(user_message)
        if self._is_override(user_message):
            # Replace the immediately preceding explicit preference, retaining the
            # category and any independent constraints already confirmed.
            previous = state.get("last_slot")
            if isinstance(previous, str) and previous != "category":
                slots.pop(previous, None)

        for clause in clauses:
            slot = self._slot_for(clause)
            existing = slots.get(slot, [])
            if not isinstance(existing, list):
                existing = []
            if clause not in existing:
                existing.append(clause)
            slots[slot] = existing
            state["last_slot"] = slot

        no_preference = re.search(r"no (?:an )?additional preference for\s+(\w+)|no preference for\s+(\w+)", user_message, re.I)
        if no_preference:
            requested = next(group for group in no_preference.groups() if group)
            slots.pop(requested.lower(), None)

        return {
            "message": "I will refine the matches as you share another product detail.",
            "ask_attribute": "other",
            "recommendations": self._recommend(state, top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
