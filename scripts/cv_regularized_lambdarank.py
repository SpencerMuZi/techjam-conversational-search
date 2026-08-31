#!/usr/bin/env python3
"""Nested, leakage-resistant CV for small regularized ranking models.

The outer folds are grouped by the complete user-profile payload, rather than
only by session id.  Hyperparameters and the early-stopping iteration are chosen
inside each outer training fold.  The untouched outer fold is evaluated once by
the real dialogue evaluator.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.retrieval import CatalogIndex


G11 = (
    "log_rating_number", "category_overlap_frac", "constraint_recip",
    "hard_miss_count", "has_price", "profile_hits", "exact_phrase_hits",
    "soft_total", "hard_total", "soft_hits", "cat_present",
)

# Query-relative evidence generalises better than raw BM25 magnitudes and
# product-popularity values, both of which can act like identifiers on 200 rows.
RANK_12 = (
    "hard_hit_rate", "hard_miss_count", "soft_hit_rate", "full_coverage",
    "exact_phrase_hits", "category_overlap_frac", "kw_recip", "cat_recip",
    "constraint_recip", "precise_prior", "route_presence_count", "budget_match",
)

FEATURE_SETS = {"g11": G11, "rank12": RANK_12}

CONFIGS = (
    {
        "name": "tiny",
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "reg_alpha": 1.0,
        "reg_lambda": 20.0,
        "min_split_gain": 0.05,
        "feature_fraction": 0.8,
        "n_hard_neg": 16,
        "n_rand_neg": 4,
    },
    {
        "name": "small",
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 20,
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "min_split_gain": 0.02,
        "feature_fraction": 0.9,
        "n_hard_neg": 24,
        "n_rand_neg": 8,
    },
)


def profile_group_ids(samples: list[dict]) -> np.ndarray:
    """Stable integer group per complete public user profile."""
    keys = [
        json.dumps(sample.get("user_profile", {}), sort_keys=True, separators=(",", ":"))
        for sample in samples
    ]
    mapping = {key: idx for idx, key in enumerate(sorted(set(keys)))}
    return np.asarray([mapping[key] for key in keys], dtype=np.int64)


def scenario_labels(data, n_sessions: int) -> np.ndarray:
    labels = np.empty(n_sessions, dtype=object)
    for session_idx, scenario in zip(data.session_idx, data.scenario):
        labels[session_idx] = scenario
    return labels


def grouped_folds(
    scenarios: np.ndarray,
    profile_groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    sessions = np.arange(len(scenarios))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(sessions, scenarios, groups=profile_groups))
    for train_sessions, test_sessions in folds:
        overlap = set(profile_groups[train_sessions]) & set(profile_groups[test_sessions])
        if overlap:
            raise AssertionError(f"profile leakage across fold: {sorted(overlap)}")
    return folds


def _mask(n_sessions: int, sessions: np.ndarray) -> np.ndarray:
    result = np.zeros(n_sessions, dtype=bool)
    result[sessions] = True
    return result


def _ordered_rows(data, rows: np.ndarray, cols: np.ndarray):
    order = np.argsort(data.group_id[rows], kind="stable")
    rows = rows[order]
    groups = data.group_id[rows]
    _, counts = np.unique(groups, return_counts=True)
    X = data.X[np.ix_(rows, cols)]
    y = data.y[rows]
    if not np.isfinite(X).all():
        raise ValueError("non-finite candidate feature detected")
    return X, y, counts


class SubsetBooster:
    """Adapt a subset-trained booster to the runtime's full feature vector."""

    trainable = False

    def __init__(self, booster, columns: np.ndarray, feature_names: tuple[str, ...]):
        self.booster = booster
        self.columns = columns
        self.feature_names = list(feature_names)

    def score(self, feature_rows):
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        return self.booster.predict(rows[:, self.columns])


class StableLogistic:
    """Small linear baseline using the proven g11 feature subset."""

    trainable = False

    def __init__(self, columns: np.ndarray, scaler, model):
        self.columns = columns
        self.scaler = scaler
        self.model = model
        self.feature_names = list(G11)

    def score(self, feature_rows):
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        return self.model.decision_function(
            self.scaler.transform(rows[:, self.columns])
        )


def fit_logistic(data, sessions: np.ndarray, seed: int) -> StableLogistic:
    n_sessions = int(data.session_idx.max()) + 1
    rows = sample_training_rows(data, _mask(n_sessions, sessions), seed=seed)
    cols = np.asarray([FEATURE_NAMES.index(name) for name in G11])
    X = data.X[np.ix_(rows, cols)]
    if not np.isfinite(X).all():
        raise ValueError("non-finite candidate feature detected")
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        solver="liblinear",
    ).fit(scaler.transform(X), data.y[rows])
    return StableLogistic(cols, scaler, model)


def fit_ranker(
    data,
    train_sessions: np.ndarray,
    feature_names: tuple[str, ...],
    config: dict,
    seed: int,
    valid_sessions: np.ndarray | None = None,
    n_estimators: int | None = None,
) -> tuple[SubsetBooster, int]:
    n_sessions = int(data.session_idx.max()) + 1
    cols = np.asarray([FEATURE_NAMES.index(name) for name in feature_names])
    train_rows = sample_training_rows(
        data,
        _mask(n_sessions, train_sessions),
        n_hard_neg=config["n_hard_neg"],
        n_rand_neg=config["n_rand_neg"],
        seed=seed,
    )
    X_train, y_train, train_counts = _ordered_rows(data, train_rows, cols)
    params = {
        key: value
        for key, value in config.items()
        if key not in {"name", "n_hard_neg", "n_rand_neg"}
    }
    if n_estimators is not None:
        params["n_estimators"] = max(1, int(n_estimators))
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        lambdarank_truncation_level=10,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
        **params,
    )
    fit_kwargs = {}
    if valid_sessions is not None:
        valid_rows = sample_training_rows(
            data,
            _mask(n_sessions, valid_sessions),
            n_hard_neg=10**9,
            n_rand_neg=0,
            seed=seed,
        )
        X_valid, y_valid, valid_counts = _ordered_rows(data, valid_rows, cols)
        fit_kwargs = {
            "eval_set": [(X_valid, y_valid)],
            "eval_group": [valid_counts],
            "eval_at": [1, 10],
            "callbacks": [lgb.early_stopping(75, first_metric_only=True, verbose=False)],
        }
    model.fit(X_train, y_train, group=train_counts, **fit_kwargs)
    best_iteration = int(model.best_iteration_ or params["n_estimators"])
    return SubsetBooster(model.booster_, cols, feature_names), best_iteration


def evaluate_sessions(index, reranker, sessions, samples, cids, cats, products):
    agent = _agent_with_index(index)
    agent.retriever.capture = False
    agent.retriever.reranker = reranker
    return evaluate(agent, [samples[i] for i in sessions], cids, cats, products)


@dataclass
class FoldResult:
    fold: int
    model: str
    hit: float
    mrr: float
    mttc: float
    technical_score: float
    selected: str = ""
    best_iteration: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    index = CatalogIndex(args.catalog)
    cids, cats, products = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index)
    scenarios = scenario_labels(data, len(samples))
    profiles = profile_group_ids(samples)
    outer_folds = grouped_folds(scenarios, profiles, args.splits, args.seed)
    results: list[FoldResult] = []

    for fold, (outer_train, outer_test) in enumerate(outer_folds):
        inner_splitter = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=args.seed + fold
        )
        inner_train_pos, inner_valid_pos = next(
            inner_splitter.split(
                outer_train,
                scenarios[outer_train],
                groups=profiles[outer_train],
            )
        )
        inner_train = outer_train[inner_train_pos]
        inner_valid = outer_train[inner_valid_pos]

        best = None
        for feature_set, feature_names in FEATURE_SETS.items():
            for config in CONFIGS:
                reranker, best_iteration = fit_ranker(
                    data,
                    inner_train,
                    feature_names,
                    config,
                    args.seed + fold,
                    valid_sessions=inner_valid,
                )
                inner_result = evaluate_sessions(
                    index, reranker, inner_valid, samples, cids, cats, products
                )
                candidate = (
                    inner_result["recommended_technical_score"],
                    feature_set,
                    config,
                    best_iteration,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
                print(json.dumps({
                    "fold": fold,
                    "stage": "inner",
                    "candidate": f"{config['name']}_{feature_set}",
                    "best_iteration": best_iteration,
                    "technical_score": inner_result["recommended_technical_score"],
                }), flush=True)

        _, feature_set, config, best_iteration = best
        selected = f"{config['name']}_{feature_set}"
        reranker, _ = fit_ranker(
            data,
            outer_train,
            FEATURE_SETS[feature_set],
            config,
            args.seed + fold,
            n_estimators=best_iteration,
        )
        outer_result = evaluate_sessions(
            index, reranker, outer_test, samples, cids, cats, products
        )
        results.append(FoldResult(
            fold=fold,
            model="regularized_lambdarank",
            hit=outer_result["hit_rate_at_10"],
            mrr=outer_result["mrr"],
            mttc=outer_result["mttc"],
            technical_score=outer_result["recommended_technical_score"],
            selected=selected,
            best_iteration=best_iteration,
        ))

        logistic = fit_logistic(data, outer_train, args.seed + fold)
        logistic_result = evaluate_sessions(
            index, logistic, outer_test, samples, cids, cats, products
        )
        results.append(FoldResult(
            fold=fold,
            model="g11_logistic",
            hit=logistic_result["hit_rate_at_10"],
            mrr=logistic_result["mrr"],
            mttc=logistic_result["mttc"],
            technical_score=logistic_result["recommended_technical_score"],
        ))
        print(json.dumps({"outer": [results[-2].__dict__, results[-1].__dict__]}), flush=True)

    summary = {}
    for model_name in sorted({row.model for row in results}):
        rows = [row for row in results if row.model == model_name]
        summary[model_name] = {
            key: {
                "mean": float(np.mean([getattr(row, key) for row in rows])),
                "std": float(np.std([getattr(row, key) for row in rows])),
                "per_fold": [float(getattr(row, key)) for row in rows],
            }
            for key in ("hit", "mrr", "mttc", "technical_score")
        }
    payload = {
        "protocol": "nested 5-fold, scenario-stratified and user-profile-grouped",
        "unique_profiles": int(len(set(profiles.tolist()))),
        "folds": [row.__dict__ for row in results],
        "summary": summary,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
