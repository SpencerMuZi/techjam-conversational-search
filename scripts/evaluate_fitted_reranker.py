#!/usr/bin/env python3
"""Fit one experimental reranker and score it with the official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from shopping_copilot.rerankers import build_reranker
from shopping_copilot.retrieval import CatalogIndex


class TraceAgent:
    """Record score margins and target ranks without changing evaluator behavior."""

    def __init__(self, agent, samples) -> None:
        self.agent = agent
        self.samples = samples
        self.sample_index = -1
        self.current_sample = None
        self.traces: dict[str, list[dict]] = {}

    def reset(self, session_id, user_profile) -> None:
        self.sample_index += 1
        self.current_sample = self.samples[self.sample_index]
        self.traces[self.current_sample["sample_id"]] = []
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k):
        response = self.agent.respond(session_id, user_message, turn, top_k)
        recommendations = response.get("recommendations") or []
        target = self.current_sample["ground_truth"]["parent_asin"]
        ids = [row.get("parent_asin") for row in recommendations]
        scores = [float(row.get("score", 0.0)) for row in recommendations]
        trace = {
            "turn": turn,
            "user_message": user_message,
            "ask_attribute": response.get("ask_attribute"),
            "target_rank": ids.index(target) + 1 if target in ids else None,
            "scores": scores,
            "margin_1_2": scores[0] - scores[1] if len(scores) > 1 else None,
        }
        retriever = self.agent.retriever
        if retriever.capture and retriever.last_candidates:
            context = retriever.last_context
            trace["context"] = {
                "route": context.route,
                "category": context.category,
                "hard_slots": context.hard_slots,
                "soft_slots": context.soft_slots,
            }
            pool_ids = [asin for asin, _ in retriever.last_candidates]
            trace["pool_size"] = len(pool_ids)
            if target in pool_ids:
                pool_scores = retriever.reranker.score(
                    [row for _, row in retriever.last_candidates]
                )
                order = sorted(range(len(pool_ids)), key=lambda i: pool_scores[i], reverse=True)
                trace["pool_target_rank"] = order.index(pool_ids.index(target)) + 1
                trace["target_score"] = float(pool_scores[pool_ids.index(target)])
            else:
                trace["pool_target_rank"] = None
        self.traces[self.current_sample["sample_id"]].append(trace)
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("logistic", "forest", "lambdamart", "all"))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--model-out", default="")
    args = parser.parse_args()

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index)
    session_mask = np.ones(int(data.session_idx.max()) + 1, dtype=bool)
    rows = sample_training_rows(data, session_mask, seed=args.seed)

    methods = ("logistic", "forest", "lambdamart") if args.method == "all" else (args.method,)
    for method in methods:
        reranker = build_reranker(method, random_state=args.seed)
        if method == "lambdamart":
            reranker.fit(data.X[rows], data.y[rows], data.group_id[rows])
        else:
            reranker.fit(data.X[rows], data.y[rows])

        agent = _agent_with_index(index)
        agent.retriever.capture = False
        agent.retriever.reranker = reranker
        traced_agent = TraceAgent(agent, samples)
        result = evaluate(traced_agent, samples, catalog_ids, categories, products)
        result["score_traces"] = traced_agent.traces
        print(f"\n[{method}]")
        for key in (
            "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"
        ):
            print(f"{key}: {result[key]}")
        print("scenario_metrics:", result["scenario_metrics"])
        if args.output:
            output = Path(args.output)
            if len(methods) > 1:
                output = output.with_name(f"{output.stem}-{method}{output.suffix}")
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.model_out and method == "lambdamart":
            reranker.model.booster_.save_model(args.model_out)


if __name__ == "__main__":
    main()
