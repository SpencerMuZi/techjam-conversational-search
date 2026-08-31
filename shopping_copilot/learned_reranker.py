"""Packaged learned rerankers with a deterministic linear fallback.

``experiments/train_reranker.py`` fits a scikit-learn ``LogisticRegression`` on
the public sessions and serialises the standardiser statistics + linear weights
to ``reranker_lr.json``. This stable linear model is the default. The higher
public-score LightGBM models remain explicit, opt-in experiments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .features import FEATURE_NAMES

DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("reranker_lr.json")
DEFAULT_LGBM_PATH = Path(__file__).with_name("reranker_lgbm.txt")
DEFAULT_WIDE_LGBM_PATH = Path(__file__).with_name("reranker_wide_lgbm.txt")


class PackagedLambdaRankReranker:
    """Load the submitted LambdaRank booster used for high-accuracy inference."""

    trainable = False

    def __init__(self, booster, variant: str = "precise") -> None:
        self.booster = booster
        self.feature_names = list(FEATURE_NAMES)
        self.meta = {
            "model": "lightgbm_lambdarank",
            "variant": variant,
            "features": len(FEATURE_NAMES),
            "pool_depth": 300 if variant == "wide" else None,
        }

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LGBM_PATH, variant: str = "precise"):
        path = Path(path)
        if not path.exists():
            return None
        try:
            import lightgbm as lgb
        except (ImportError, OSError):
            return None
        return cls(lgb.Booster(model_file=str(path)), variant=variant)

    def score(self, feature_rows):
        import numpy as np

        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        return self.booster.predict(rows)


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
    # The small linear model has a negligible train/CV gap.  High-capacity
    # LambdaRank models remain explicit public-score experiments; making one of
    # them the implicit default would silently trade generalisation for a score
    # measured on the same 200 sessions used for fitting.
    mode = os.environ.get("SHOPPING_COPILOT_RERANKER", "logistic").lower()
    if mode == "manual":
        return None
    if mode in {"wide", "wide_lambdarank"}:
        reranker = PackagedLambdaRankReranker.load(
            DEFAULT_WIDE_LGBM_PATH, variant="wide"
        )
        if reranker is not None:
            return reranker
    if mode in {"lambdarank", "lgbm", "wide", "wide_lambdarank"}:
        reranker = PackagedLambdaRankReranker.load()
        if reranker is not None:
            return reranker
    return PackagedLogisticReranker.load()
