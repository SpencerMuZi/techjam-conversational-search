"""Interchangeable reranking strategies over the shared feature vector.

Every reranker exposes ``score(feature_rows) -> sequence[float]`` (higher ranks
earlier). The learned ones also expose ``fit(X, y, groups)``.

Only ``ManualReranker`` is import-safe without scientific-Python; the logistic /
forest / LambdaRank builders lazily import scikit-learn / LightGBM, so this module
can be imported (and ``ManualReranker`` used in tests) with the standard library
alone.
"""
from __future__ import annotations

import math

from .features import FEATURE_NAMES

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class ManualReranker:
    """The hand-tuned structured score, expressed over the feature vector.

    Kept bit-identical to ``HybridRetriever._structured_score`` + the fused RRF
    term so it is a faithful baseline in the model comparison. Pure Python.
    """

    trainable = False

    def score(self, feature_rows) -> list[float]:
        f = _IDX
        out: list[float] = []
        for r in feature_rows:
            s = r[f["rrf_fused_score"]]
            s += 1.3 * r[f["precise_prior"]]
            s += 0.8 * r[f["category_overlap_frac"]]
            s += 1.6 * r[f["hard_hits"]] + 0.35 * r[f["hard_idf_sum"]] - 0.7 * r[f["hard_miss_count"]]
            s += 0.4 * r[f["soft_hits"]] + 0.15 * r[f["soft_idf_sum"]]
            s += 0.025 * r[f["profile_hits_capped"]]
            s += min(r[f["log_rating_number"]] / 20.0, 0.08)
            s += min(r[f["avg_rating"]] / 5.0, 1.0) * 0.025
            out.append(s)
        return out


class _SklearnProbaReranker:
    trainable = True

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, groups=None):
        self.estimator.fit(X, y)
        return self

    def score(self, feature_rows):
        import numpy as np

        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        return self.estimator.predict_proba(rows)[:, 1]


def logistic_reranker(random_state: int = 0) -> _SklearnProbaReranker:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000, random_state=random_state
        )),
    ])
    return _SklearnProbaReranker(pipe)


def forest_reranker(random_state: int = 0) -> _SklearnProbaReranker:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=1,
        random_state=random_state,
    )
    return _SklearnProbaReranker(model)


class LambdaReranker:
    """LightGBM LambdaRank - pairwise/listwise objective aligned with MRR."""

    trainable = True

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.model = None

    def fit(self, X, y, groups):
        import lightgbm as lgb
        import numpy as np

        order = np.argsort(groups, kind="stable")
        X, y, groups = np.asarray(X)[order], np.asarray(y)[order], np.asarray(groups)[order]
        _, counts = np.unique(groups, return_counts=True)
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            label_gain=[0, 1],
            random_state=self.random_state,
            n_jobs=1,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        self.model.fit(X, y, group=counts)
        return self

    def score(self, feature_rows):
        import numpy as np

        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        # Bypass the sklearn wrapper at inference: the booster consumes the same
        # numeric matrix without emitting one feature-name warning per turn.
        return self.model.booster_.predict(rows)


def build_reranker(name: str, random_state: int = 0):
    if name == "manual":
        return ManualReranker()
    if name == "logistic":
        return logistic_reranker(random_state)
    if name == "forest":
        return forest_reranker(random_state)
    if name == "lambdamart":
        return LambdaReranker(random_state)
    raise ValueError(f"unknown reranker: {name}")
