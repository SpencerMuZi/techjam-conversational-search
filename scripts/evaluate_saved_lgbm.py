#!/usr/bin/env python3
"""Evaluate a saved LightGBM booster without rebuilding the training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.dataset import _agent_with_index
from scripts.evaluate_fitted_reranker import TraceAgent
from shopping_copilot.retrieval import CatalogIndex


class BoosterReranker:
    def __init__(self, model_path: str) -> None:
        self.booster = lgb.Booster(model_file=model_path)

    def score(self, feature_rows):
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        return self.booster.predict(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-id", default="")
    args = parser.parse_args()

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    if args.sample_id:
        samples = [sample for sample in samples if sample["sample_id"] == args.sample_id]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = _agent_with_index(index)
    agent.retriever.capture = True
    agent.retriever.reranker = BoosterReranker(args.model)
    traced_agent = TraceAgent(agent, samples)
    result = evaluate(traced_agent, samples, catalog_ids, categories, products)
    result["score_traces"] = traced_agent.traces
    summary = {
        key: value for key, value in result.items()
        if key not in ("sessions", "score_traces")
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
