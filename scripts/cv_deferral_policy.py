#!/usr/bin/env python3
"""Profile-grouped CV for first-turn recommendation deferral policies."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluator.local_evaluator import catalog_index, load_jsonl
from experiments.dataset import build_dataset
from scripts.cv_regularized_lambdarank import (
    evaluate_sessions,
    fit_logistic,
    grouped_folds,
    profile_group_ids,
    scenario_labels,
)
from shopping_copilot.retrieval import CatalogIndex


POLICIES = ("none", "all", "adaptive", "browsing")


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
    folds = grouped_folds(scenarios, profiles, args.splits, args.seed)
    results = {policy: [] for policy in POLICIES}

    previous = os.environ.get("SHOPPING_COPILOT_DEFERRAL")
    try:
        for fold, (train_sessions, test_sessions) in enumerate(folds):
            reranker = fit_logistic(data, train_sessions, args.seed + fold)
            for policy in POLICIES:
                os.environ["SHOPPING_COPILOT_DEFERRAL"] = policy
                result = evaluate_sessions(
                    index,
                    reranker,
                    test_sessions,
                    samples,
                    cids,
                    cats,
                    products,
                )
                row = {
                    "fold": fold,
                    "hit": result["hit_rate_at_10"],
                    "mrr": result["mrr"],
                    "mttc": result["mttc"],
                    "technical_score": result["recommended_technical_score"],
                }
                results[policy].append(row)
                print(json.dumps({"policy": policy, **row}), flush=True)
    finally:
        if previous is None:
            os.environ.pop("SHOPPING_COPILOT_DEFERRAL", None)
        else:
            os.environ["SHOPPING_COPILOT_DEFERRAL"] = previous

    summary = {}
    for policy, rows in results.items():
        summary[policy] = {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows])),
                "per_fold": [float(row[key]) for row in rows],
            }
            for key in ("hit", "mrr", "mttc", "technical_score")
        }
    payload = {
        "protocol": "5-fold, scenario-stratified and user-profile-grouped",
        "unique_profiles": int(len(set(profiles.tolist()))),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
