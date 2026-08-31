"""V4 submission entry point: V3 retrieval plus optional XGBoost reranking."""
from __future__ import annotations

import json
import math
from pathlib import Path

from v3_core import Agent as V3Agent

try:
    from xgboost import XGBRanker
except ImportError:  # Official scoring can still run the V3 fallback offline.
    XGBRanker = None


class Agent(V3Agent):
    """V3 with a trained candidate reranker and a dependency-safe fallback.

    The required model file is colocated with this module. If either XGBoost or
    the model file is unavailable, every response uses the unchanged V3 ranking.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", model_path: str | Path | None = None) -> None:
        super().__init__(catalog_path)
        self._products = self._load_products()
        self._reranker = self._load_reranker(model_path)

    def _load_products(self) -> dict[str, dict]:
        products: dict[str, dict] = {}
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                products[str(product["parent_asin"])] = product
        return products

    @staticmethod
    def _product_text(product: dict, fields: tuple[str, ...]) -> str:
        values: list[str] = []
        for field in fields:
            value = product.get(field)
            if isinstance(value, dict):
                values.extend(f"{key} {item}" for key, item in value.items())
            elif isinstance(value, list):
                values.extend(map(str, value))
            elif value is not None:
                values.append(str(value))
        return " ".join(values).lower()

    def _load_reranker(self, model_path: str | Path | None):
        if XGBRanker is None:
            return None
        path = Path(model_path) if model_path else Path(__file__).with_name("xgboost_reranker.json")
        if not path.is_file():
            return None
        model = XGBRanker()
        model.load_model(path)
        return model

    def _features(self, state: dict[str, object], product: dict, v3_score: float, rank: int, turn: int) -> list[float]:
        slots = state["slots"]
        if not isinstance(slots, dict):
            return []
        values = self._slot_text(slots)
        full = self._product_text(product, ("title", "categories", "features", "details", "store", "description"))
        title = self._product_text(product, ("title",))
        category = self._product_text(product, ("categories",))
        slot_names = ("material", "color", "size", "brand", "budget", "style", "use_case", "feature")
        slot_cover = [self._coverage(full, slots.get(name, [])) for name in slot_names]
        rating = float(product.get("average_rating") or 0.0) / 5.0
        rating_count = math.log1p(float(product.get("rating_number") or 0.0)) / 15.0
        return [
            float(v3_score), 1.0 / rank, self._coverage(full, values),
            self._coverage(title, values), self._coverage(category, values),
            self._budget_fit(str(product.get("price") or ""), values),
            self._profile_affinity(full, state.get("profile")), rating, rating_count,
            float(product.get("price") not in (None, "")), turn / 10.0,
            min(1.0, len(values) / 8.0), *slot_cover,
        ]

    def _rank(self, state: dict[str, object], candidates: list[dict], turn: int, top_k: int) -> list[dict]:
        if self._reranker is None:
            return candidates[:top_k]
        records = [(rank, item, self._products.get(str(item["parent_asin"]))) for rank, item in enumerate(candidates, 1)]
        if any(product is None for _, _, product in records):
            return candidates[:top_k]
        features = [self._features(state, product, float(item.get("score", 0.0)), rank, turn)
                    for rank, item, product in records]
        try:
            prediction = self._reranker.predict(features)
        except Exception:
            return candidates[:top_k]
        ranked = [item for _, item in sorted(zip(prediction, candidates), key=lambda pair: -pair[0])]
        return ranked[:top_k]

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
            previous = state.get("last_slot")
            if isinstance(previous, str) and previous != "category":
                slots.pop(previous, None)
        for clause in clauses:
            slot = self._slot_for(clause)
            current = slots.get(slot, [])
            if not isinstance(current, list):
                current = []
            if clause not in current:
                current.append(clause)
            slots[slot] = current
            state["last_slot"] = slot

        candidates = super()._recommend(state, 300)
        return {
            "message": "I will refine the matches as you share another product detail.",
            "ask_attribute": "other",
            "recommendations": self._rank(state, candidates, turn, top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
