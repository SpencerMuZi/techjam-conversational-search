"""Replay the 200 public dev sessions and capture per-candidate rerank features.

One row per (session, turn, candidate). The dialogue is driven by the *official*
evaluator's deterministic customer policy, so the captured pools are exactly what
the live agent would rerank.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from shopping_copilot.agent import ShoppingCopilotAgent
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.retrieval import CatalogIndex


@dataclass
class RerankDataset:
    X: np.ndarray            # (n_rows, n_features)
    y: np.ndarray            # (n_rows,) 1 == target
    group_id: np.ndarray     # (n_rows,) integer id per (session, turn)
    session_idx: np.ndarray  # (n_rows,) 0..199, for GroupKFold by session
    scenario: np.ndarray     # (n_rows,) scenario string per row
    feature_names: list[str]
    is_mined: np.ndarray = None  # (n_rows,) 1 == catalog-mined confusable negative

    def __post_init__(self):
        if self.is_mined is None:
            self.is_mined = np.zeros(len(self.y), dtype=np.int64)

    def groups_with_positive(self) -> int:
        pos_groups = {g for g, label in zip(self.group_id, self.y) if label == 1}
        return len(pos_groups)


def _shared_index(catalog_path: str) -> CatalogIndex:
    return CatalogIndex(catalog_path)


def _agent_with_index(index: CatalogIndex) -> ShoppingCopilotAgent:
    from shopping_copilot.clarification import ClarificationPolicy
    from shopping_copilot.config import AgentConfig
    from shopping_copilot.context import ContextBuilder
    from shopping_copilot.retrieval import HybridRetriever
    from shopping_copilot.semantic import NullSemanticRetriever
    from shopping_copilot.slots import SlotExtractor
    from shopping_copilot.state import SessionStore

    agent = ShoppingCopilotAgent.__new__(ShoppingCopilotAgent)
    agent.config = AgentConfig()
    agent.sessions = SessionStore()
    agent.extractor = SlotExtractor()
    agent.context_builder = ContextBuilder()
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.config = agent.config
    retriever.index = index
    retriever.semantic = NullSemanticRetriever()
    retriever.reranker = None
    retriever.capture = True
    retriever.last_candidates = []
    agent.retriever = retriever
    agent.clarification = ClarificationPolicy(agent.config)
    return agent


def _mine_confusables(index: CatalogIndex, target: str, k: int, exclude: set[str]) -> list[str]:
    """Catalog items whose own text most resembles the target's - the confusables
    retrieval might miss. Used only to harden the training negatives."""
    if k <= 0 or target not in index.documents:
        return []
    hits = index.search_fts_scored(index.documents[target].searchable_text, k + len(exclude) + 5)
    out: list[str] = []
    for asin, _ in hits:
        if asin != target and asin not in exclude:
            out.append(asin)
        if len(out) >= k:
            break
    return out


def build_dataset(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
    index: CatalogIndex | None = None,
    mine_k: int = 0,
) -> RerankDataset:
    from shopping_copilot.features import candidate_features

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    index = index or _shared_index(catalog_path)
    agent = _agent_with_index(index)

    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    session_ids: list[int] = []
    scenarios: list[str] = []
    mined_flags: list[int] = []
    group_counter = 0

    for session_idx, sample in enumerate(samples):
        session_id = f"cv_{session_idx}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            pool = agent.retriever.last_candidates
            pool_ids = {asin for asin, _ in pool}
            for asin, vector in pool:
                rows.append(vector)
                labels.append(1 if asin == target else 0)
                group_ids.append(group_counter)
                session_ids.append(session_idx)
                scenarios.append(sample["scenario_type"])
                mined_flags.append(0)
            if mine_k and target in pool_ids:
                ctx, sig = agent.retriever.last_context, agent.retriever.last_signals
                for asin in _mine_confusables(index, target, mine_k, pool_ids):
                    rows.append(candidate_features(index, index.documents[asin], ctx, sig))
                    labels.append(0)
                    group_ids.append(group_counter)
                    session_ids.append(session_idx)
                    scenarios.append(sample["scenario_type"])
                    mined_flags.append(1)
            group_counter += 1

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

    return RerankDataset(
        X=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        group_id=np.asarray(group_ids, dtype=np.int64),
        session_idx=np.asarray(session_ids, dtype=np.int64),
        scenario=np.asarray(scenarios, dtype=object),
        feature_names=list(FEATURE_NAMES),
        is_mined=np.asarray(mined_flags, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the captured rerank dataset")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    data = build_dataset(args.catalog, args.dataset)
    n_groups = int(data.group_id.max()) + 1
    pos = int(data.y.sum())
    print(f"rows              : {len(data.y):,}")
    print(f"features          : {data.X.shape[1]}")
    print(f"groups (sess,turn): {n_groups}")
    print(f"positives         : {pos}")
    print(f"groups w/ positive: {data.groups_with_positive()} / {n_groups}"
          f"  ({data.groups_with_positive() / n_groups:.1%} of turns have the target in-pool)")
    print(f"neg:pos ratio     : {(len(data.y) - pos) / max(pos, 1):.1f} : 1")
    print(f"pool size / turn  : mean {len(data.y) / n_groups:.0f}")

    import collections
    by_scen = collections.Counter(data.scenario)
    pos_by_scen = collections.Counter(s for s, label in zip(data.scenario, data.y) if label == 1)
    print("\nper scenario   rows / target-in-pool turns:")
    for scen in sorted(by_scen):
        gids = {g for g, s in zip(data.group_id, data.scenario) if s == scen}
        pgids = {g for g, s, label in zip(data.group_id, data.scenario, data.y) if s == scen and label == 1}
        print(f"  {scen:16s} rows={by_scen[scen]:6d}  turns={len(gids):4d}  with-target={len(pgids):4d}  ({len(pgids)/max(len(gids),1):.0%})")

    print("\nfeature ranges (min / mean / max):")
    for i, name in enumerate(data.feature_names):
        col = data.X[:, i]
        print(f"  {name:22s} {col.min():9.3f} {col.mean():9.3f} {col.max():9.3f}")


if __name__ == "__main__":
    main()
