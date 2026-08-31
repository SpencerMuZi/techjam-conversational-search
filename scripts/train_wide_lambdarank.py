#!/usr/bin/env python3
"""Train the packaged 300-candidate LambdaRank model on public sessions."""

from __future__ import annotations

import argparse

import numpy as np

from experiments.cv import sample_training_rows
from experiments.dataset import build_dataset
from scripts.cv_lambdarank_capacity import WIDE_POOL_CONFIG
from scripts.tune_lambdarank import fit_ranker
from shopping_copilot.config import AgentConfig
from shopping_copilot.retrieval import CatalogIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model-out", default="shopping_copilot/reranker_wide_lgbm.txt")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pool_depth = 300
    agent_config = AgentConfig(
        retrieval_depth=pool_depth,
        rerank_depth=pool_depth,
        precise_seed=pool_depth,
        early_pool_enabled=False,
    )
    index = CatalogIndex(args.catalog)
    data = build_dataset(
        args.catalog,
        args.dataset,
        index=index,
        config=agent_config,
    )
    session_mask = np.ones(int(data.session_idx.max()) + 1, dtype=bool)
    rows = sample_training_rows(
        data,
        session_mask,
        n_hard_neg=WIDE_POOL_CONFIG["n_hard_neg"],
        n_rand_neg=WIDE_POOL_CONFIG["n_rand_neg"],
        seed=args.seed,
    )
    model = fit_ranker(
        data.X[rows], data.y[rows], data.group_id[rows], WIDE_POOL_CONFIG, args.seed
    )
    model.booster_.save_model(args.model_out)
    print(
        f"saved {args.model_out}: rows={len(rows):,}, "
        f"positives={int(data.y[rows].sum()):,}"
    )


if __name__ == "__main__":
    main()
