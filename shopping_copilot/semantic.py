from __future__ import annotations


class NullSemanticRetriever:
    """No-op adapter used until a local dense model is configured.

    Keeping this behind a small interface lets the production pipeline add a
    Sentence Transformer or another in-memory encoder without changing the
    dialogue policy or the official Agent entry point.
    """

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        return []
