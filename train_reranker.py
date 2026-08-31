"""Leakage-safe 5-fold test of an XGBoost reranker on V3 candidates.

Only target ASIN is used as the training label. Features use runtime-safe
inputs: dialogue slots, profile, frozen catalog, and V3 candidate score.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "techjam-conversational-search-participant-kit" / "techjam-conversational-search-participant-kit"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, catalog_index, customer_reply, initial_message, load_jsonl,
    materialize_hidden_fields, coarse_category,
)


def load_catalog(path: Path) -> dict[str, dict]:
    return {str(row["parent_asin"]): row for row in load_jsonl(path)}


def product_text(product: dict, fields: tuple[str, ...]) -> str:
    values = []
    for field in fields:
        value = product.get(field)
        if isinstance(value, dict):
            values.extend(f"{k} {v}" for k, v in value.items())
        elif isinstance(value, list):
            values.extend(map(str, value))
        elif value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def make_features(agent: Agent, state: dict, product: dict, v3_score: float, rank: int, turn: int) -> list[float]:
    slots = state["slots"]
    values = agent._slot_text(slots)
    full = product_text(product, ("title", "categories", "features", "details", "store", "description"))
    title = product_text(product, ("title",))
    category = product_text(product, ("categories",))
    slot_names = ("material", "color", "size", "brand", "budget", "style", "use_case", "feature")
    slot_cover = [agent._coverage(full, slots.get(name, [])) for name in slot_names]
    rating = float(product.get("average_rating") or 0.0) / 5.0
    rating_count = math.log1p(float(product.get("rating_number") or 0.0)) / 15.0
    return [
        float(v3_score), 1.0 / rank, agent._coverage(full, values),
        agent._coverage(title, values), agent._coverage(category, values),
        agent._budget_fit(str(product.get("price") or ""), values),
        agent._profile_affinity(full, state.get("profile")), rating, rating_count,
        float(product.get("price") not in (None, "")), turn / MAX_TURNS,
        min(1.0, len(values) / 8.0), *slot_cover,
    ]


def build_contexts(agent: Agent, samples: list[dict], products: dict[str, dict], categories: dict[str, list[str]]) -> dict[str, list[dict]]:
    contexts: dict[str, list[dict]] = {}
    for sample in samples:
        session_id = f"feature_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed, boundary_used = set(), False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
        rows = []
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, 300)
            state = agent._sessions[session_id]
            candidates = []
            for rank, item in enumerate(response["recommendations"], 1):
                asin = str(item["parent_asin"])
                candidates.append({
                    "asin": asin,
                    "features": make_features(agent, state, products[asin], float(item["score"]), rank, turn),
                })
            rows.append({"target": target, "turn": turn, "candidates": candidates})
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
                user_message, boundary_used = customer_reply(effective, "other", disclosed, boundary_used)
        contexts[sample["sample_id"]] = rows
    return contexts


def fit_ranker(contexts: dict[str, list[dict]], ids: list[str]) -> XGBRanker:
    features, labels, groups = [], [], []
    for sample_id in ids:
        for context in contexts[sample_id]:
            label = [int(item["asin"] == context["target"]) for item in context["candidates"]]
            if not any(label):
                continue
            features.extend(item["features"] for item in context["candidates"])
            labels.extend(label)
            groups.append(len(label))
    model = XGBRanker(
        objective="rank:pairwise", n_estimators=120, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.85, reg_lambda=4.0, min_child_weight=4,
        n_jobs=4, random_state=2026,
    )
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels), group=np.asarray(groups))
    return model


def score(contexts: dict[str, list[dict]], ids: list[str], model: XGBRanker | None) -> dict:
    hits, reciprocal, turns = [], [], []
    for sample_id in ids:
        hit_turn, rank_at_hit = None, None
        for context in contexts[sample_id]:
            candidates = context["candidates"]
            if model is None:
                ordered = sorted(candidates, key=lambda x: -x["features"][0])
            else:
                prediction = model.predict(np.asarray([x["features"] for x in candidates], dtype=np.float32))
                ordered = [item for _, item in sorted(zip(prediction, candidates), key=lambda x: -x[0])]
            ranked = [item["asin"] for item in ordered[:10]]
            if context["target"] in ranked:
                hit_turn, rank_at_hit = context["turn"], ranked.index(context["target"]) + 1
                break
        hits.append(hit_turn is not None)
        reciprocal.append(0.0 if rank_at_hit is None else 1.0 / rank_at_hit)
        turns.append(11 if hit_turn is None else hit_turn)
    mttc = float(np.mean(turns))
    hit_rate, mrr = float(np.mean(hits)), float(np.mean(reciprocal))
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "hit_rate_at_10": round(hit_rate, 6), "mrr": round(mrr, 6), "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency, 6),
    }


def main() -> None:
    catalog_path = ROOT / "catalog.jsonl"
    samples = load_jsonl(KIT / "data" / "public_set.jsonl")
    _, categories, _ = catalog_index(catalog_path)
    products = load_catalog(catalog_path)
    agent = Agent(catalog_path)
    contexts = build_contexts(agent, samples, products, categories)
    ids = [sample["sample_id"] for sample in samples]
    folds = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=2026)
    for fold, (train_index, valid_index) in enumerate(splitter.split(ids), 1):
        train_ids = [ids[i] for i in train_index]
        valid_ids = [ids[i] for i in valid_index]
        model = fit_ranker(contexts, train_ids)
        folds.append({"fold": fold, "v3": score(contexts, valid_ids, None), "xgboost": score(contexts, valid_ids, model)})
        print(f"fold {fold}: {folds[-1]['v3']} -> {folds[-1]['xgboost']}")
    mean = lambda key, variant: round(float(np.mean([fold[variant][key] for fold in folds])), 6)
    result = {
        "protocol": "5-fold KFold by session_id; no session appears in both train and validation",
        "candidate_source": "V3 retrieves 300 candidates; target ASIN is used only as a training label",
        "folds": folds,
        "mean": {variant: {key: mean(key, variant) for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score")} for variant in ("v3", "xgboost")},
    }
    (Path(__file__).with_name("xgboost_cv_results.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
