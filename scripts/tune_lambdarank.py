#!/usr/bin/env python3
"""Tune LambdaRank capacity against the real evaluator on a fixed candidate set."""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from shopping_copilot.retrieval import CatalogIndex


CONFIGS = (
    {"name": "baseline", "num_leaves": 31, "min_child_samples": 20, "n_estimators": 400, "learning_rate": 0.05},
    {"name": "medium", "num_leaves": 63, "min_child_samples": 10, "n_estimators": 650, "learning_rate": 0.04},
    {"name": "deep", "num_leaves": 127, "min_child_samples": 5, "n_estimators": 900, "learning_rate": 0.03},
    {"name": "very_deep", "num_leaves": 255, "min_child_samples": 2, "n_estimators": 1200, "learning_rate": 0.025},
    {"name": "wide_fast", "num_leaves": 127, "min_child_samples": 2, "n_estimators": 600, "learning_rate": 0.06},
)

EXTENDED_CONFIGS = (
    {"name": "deep_more_neg", "num_leaves": 255, "min_child_samples": 2,
     "n_estimators": 1400, "learning_rate": 0.025, "n_hard_neg": 64, "n_rand_neg": 16},
    {"name": "extreme_more_neg", "num_leaves": 511, "min_child_samples": 1,
     "n_estimators": 1800, "learning_rate": 0.02, "n_hard_neg": 96, "n_rand_neg": 24},
    {"name": "extreme_all", "num_leaves": 511, "min_child_samples": 1,
     "n_estimators": 1000, "learning_rate": 0.03, "n_hard_neg": 10**9, "n_rand_neg": 0},
)


class BoosterReranker:
    def __init__(self, booster) -> None:
        self.booster = booster

    def score(self, feature_rows):
        rows = np.asarray(feature_rows, dtype=np.float64)
        return self.booster.predict(rows) if rows.size else np.zeros((0,))


def fit_ranker(X, y, groups, config, seed):
    order = np.argsort(groups, kind="stable")
    X, y, groups = X[order], y[order], groups[order]
    _, counts = np.unique(groups, return_counts=True)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        label_gain=[0, 1],
        max_depth=-1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
        **{
            key: value for key, value in config.items()
            if key not in ("name", "n_hard_neg", "n_rand_neg")
        },
    )
    model.fit(X, y, group=counts)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model-out", default="shopping_copilot/reranker_lgbm.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index)
    mask = np.ones(int(data.session_idx.max()) + 1, dtype=bool)
    best = None
    configs = EXTENDED_CONFIGS if args.extended else CONFIGS
    for config in configs:
        rows = sample_training_rows(
            data,
            mask,
            n_hard_neg=config.get("n_hard_neg", 24),
            n_rand_neg=config.get("n_rand_neg", 8),
            seed=args.seed,
        )
        model = fit_ranker(data.X[rows], data.y[rows], data.group_id[rows], config, args.seed)
        agent = _agent_with_index(index)
        agent.retriever.capture = False
        agent.retriever.reranker = BoosterReranker(model.booster_)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        summary = {
            "name": config["name"],
            "hit": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
            "training_rows": int(len(rows)),
        }
        print(json.dumps(summary), flush=True)
        if best is None or summary["technical_score"] > best[0]:
            best = (summary["technical_score"], config["name"], model.booster_)

    best[2].save_model(args.model_out)
    print(f"saved {best[1]} ({best[0]:.6f}) to {args.model_out}")


if __name__ == "__main__":
    main()
