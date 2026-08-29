"""Standard-library-only linear reranker loaded from trained coefficients.

``experiments/train_reranker.py`` fits a scikit-learn ``LogisticRegression`` on
the public sessions and serialises the standardiser statistics + linear weights
to ``reranker_lr.json``. At inference we only need ``json`` + ``math`` here, so
the shipped agent keeps no scientific-Python dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .features import FEATURE_NAMES

DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("reranker_lr.json")


class PackagedLogisticReranker:
    """Reproduces ``LogisticRegression`` decision scores with plain Python.

    ``score`` returns the raw linear logit; it is monotone in P(target), which is
    all a ranker needs, so the sigmoid is skipped.
    """

    trainable = False

    def __init__(self, payload: dict) -> None:
        # The model may use only a subset of FEATURE_NAMES. Look each stored
        # feature up in the live vector layout; keep the model's own order.
        self._columns = [FEATURE_NAMES.index(name) for name in payload["feature_names"]]
        self.mean = [float(v) for v in payload["mean"]]
        self.scale = [float(v) or 1.0 for v in payload["scale"]]
        self.coef = [float(v) for v in payload["coef"]]
        self.intercept = float(payload["intercept"])
        self.feature_names = list(payload["feature_names"])
        self.meta = {k: payload.get(k) for k in ("model", "trained_on", "n_rows", "n_positive")}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_WEIGHTS_PATH) -> "PackagedLogisticReranker | None":
        path = Path(path)
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def score(self, feature_rows) -> list[float]:
        cols, mean, scale, coef, bias = self._columns, self.mean, self.scale, self.coef, self.intercept
        width = len(coef)
        out: list[float] = []
        for row in feature_rows:
            total = bias
            for k in range(width):
                total += (row[cols[k]] - mean[k]) / scale[k] * coef[k]
            out.append(total)
        return out


def default_reranker(config_enabled: bool = True):
    """Return the packaged reranker unless disabled by config or env override."""
    if not config_enabled:
        return None
    if os.environ.get("SHOPPING_COPILOT_RERANKER", "learned").lower() == "manual":
        return None
    return PackagedLogisticReranker.load()
