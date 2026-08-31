"""Optional local cross-encoder that re-scores the linear reranker's shortlist.

Off by default. When ``AgentConfig.cross_encoder`` is set and
``sentence-transformers`` + the model weights are available, the top
``cross_encoder_depth`` candidates from the logistic reranker are re-scored by a
small MS-MARCO cross-encoder that reads the conversation text and the product
text jointly, then blended back with the linear score.

Everything here degrades gracefully: if the dependency or the model is missing,
``load_cross_encoder`` returns ``None`` and the agent keeps the linear reranker.
"""
from __future__ import annotations

import os
import re

from .models import ProductDocument, SearchContext

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_WS = re.compile(r"\s+")


def _clip(text: str, limit: int) -> str:
    return _WS.sub(" ", text).strip()[:limit]


class CrossEncoderReranker:
    def __init__(self, model, depth: int = 20, weight: float = 0.7,
                 query_chars: int = 320, doc_chars: int = 1600) -> None:
        self._model = model
        self.depth = depth
        self.weight = weight          # blend: weight*ce + (1-weight)*linear, both z-scored
        self.query_chars = query_chars
        self.doc_chars = doc_chars

    # ------------------------------------------------------------------
    def build_query(self, context: SearchContext) -> str:
        parts: list[str] = []
        if context.current_message:
            parts.append(str(context.current_message))
        if context.category:
            parts.append(str(context.category))
        seen = {p.lower() for p in parts}
        for value in (*context.hard_values, *context.soft_values):
            v = str(value).strip()
            if v and v.lower() not in seen:
                parts.append(v)
                seen.add(v.lower())
        return _clip(" ; ".join(parts), self.query_chars)

    def build_doc(self, product: ProductDocument) -> str:
        parts = [product.title, product.features, product.details, product.description]
        return _clip(" ".join(p for p in parts if p), self.doc_chars)

    # ------------------------------------------------------------------
    def rerank(self, context: SearchContext, scored: list[tuple[str, float]],
               documents) -> list[tuple[str, float]]:
        if not scored or self.weight <= 0.0:
            return scored
        head = scored[: self.depth]
        tail = scored[self.depth:]
        query = self.build_query(context)
        pairs = [[query, self.build_doc(documents[asin])] for asin, _ in head]
        ce = list(self._model.predict(pairs, show_progress_bar=False))
        lin = [s for _, s in head]

        ce_z = _zscore(ce)
        lin_z = _zscore(lin)
        w = self.weight
        blended = [
            (head[i][0], w * ce_z[i] + (1.0 - w) * lin_z[i])
            for i in range(len(head))
        ]
        blended.sort(key=lambda kv: kv[1], reverse=True)
        # keep tail ordering / scores; only the shortlist is reordered
        return blended + tail


def _zscore(values) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5 or 1.0
    return [(v - mean) / std for v in values]


def load_cross_encoder(config_enabled: bool, model_name: str = DEFAULT_MODEL,
                       depth: int = 20, weight: float = 0.7):
    """Return a ``CrossEncoderReranker`` or ``None`` (dependency/model missing,
    or disabled by config / ``SHOPPING_COPILOT_CROSS_ENCODER=0``)."""
    if not config_enabled:
        return None
    if os.environ.get("SHOPPING_COPILOT_CROSS_ENCODER", "1").lower() in ("0", "false", "off"):
        return None
    name = os.environ.get("SHOPPING_COPILOT_CROSS_ENCODER_MODEL", model_name)
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        model = CrossEncoder(name)
    except Exception:
        return None
    return CrossEncoderReranker(model, depth=depth, weight=weight)
