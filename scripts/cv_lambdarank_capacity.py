#!/usr/bin/env python3
"""Session-grouped real-evaluator CV for the submitted high-capacity ranker."""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from scripts.tune_lambdarank import BoosterReranker, EXTENDED_CONFIGS, fit_ranker
from shopping_copilot.config import AgentConfig
from shopping_copilot.retrieval import CatalogIndex


WIDE_POOL_CONFIG = {
    "name": "wide_pool",
    "num_leaves": 127,
    "min_child_samples": 2,
    "n_estimators": 800,
    "learning_rate": 0.04,
    "n_hard_neg": 10**9,
    "n_rand_neg": 0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-depth", type=int, default=0)
    parser.add_argument("--profile", choices=("extreme_all", "wide_pool"), default="extreme_all")
    args = parser.parse_args()

    config = (
        WIDE_POOL_CONFIG
        if args.profile == "wide_pool"
        else next(row for row in EXTENDED_CONFIGS if row["name"] == "extreme_all")
    )
    agent_config = None
    if args.pool_depth:
        agent_config = AgentConfig(
            retrieval_depth=args.pool_depth,
            rerank_depth=args.pool_depth,
            precise_seed=args.pool_depth,
            early_pool_enabled=False,
        )
    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index, config=agent_config)
    n_sessions = int(data.session_idx.max()) + 1
    scenarios = np.empty(n_sessions, dtype=object)
    for session_idx, scenario in zip(data.session_idx, data.scenario):
        scenarios[session_idx] = scenario

    session_ids = np.arange(n_sessions)
    splitter = StratifiedGroupKFold(
        n_splits=args.splits, shuffle=True, random_state=args.seed
    )
    folds = []
    for fold, (train_sessions, test_sessions) in enumerate(
        splitter.split(session_ids, scenarios, groups=session_ids)
    ):
        train_mask = np.zeros(n_sessions, dtype=bool)
        train_mask[train_sessions] = True
        rows = sample_training_rows(
            data,
            train_mask,
            n_hard_neg=config["n_hard_neg"],
            n_rand_neg=config["n_rand_neg"],
            seed=args.seed,
        )
        model = fit_ranker(
            data.X[rows], data.y[rows], data.group_id[rows], config, args.seed
        )
        agent = _agent_with_index(index, agent_config)
        agent.retriever.capture = False
        agent.retriever.reranker = BoosterReranker(model.booster_)
        test_samples = [samples[index] for index in test_sessions]
        result = evaluate(agent, test_samples, catalog_ids, categories, products)
        row = {
            "fold": fold,
            "hit": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
        }
        folds.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in folds])),
            "std": float(np.std([row[key] for row in folds])),
        }
        for key in ("hit", "mrr", "mttc", "technical_score")
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
