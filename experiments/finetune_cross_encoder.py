"""Fine-tune the shortlist cross-encoder on the 200 public sessions and CV it.

Per fold: fine-tune ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on the training
sessions' (conversation, product) text pairs (label = is-target), then score the
held-out sessions with the real evaluator, blended with the frozen g11 logistic
reranker. Compares against logistic-only.

    python -m pip install -r requirements-experiments.txt
    python -m experiments.finetune_cross_encoder --folds 5 --epochs 2 --weights 0.5,1.0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, evaluate,
    initial_message, load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from shopping_copilot.cross_encoder import CrossEncoderReranker, DEFAULT_MODEL
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.retrieval import CatalogIndex

G11 = json.loads(Path("shopping_copilot/reranker_lr.json").read_text(encoding="utf-8"))["feature_names"]
_FUSED = FEATURE_NAMES.index("rrf_fused_score")


# --------------------------------------------------------------------------
# text-pair capture (mirrors experiments/dataset.build_dataset)
# --------------------------------------------------------------------------
def build_text_pairs(catalog_path, dataset_path, index):
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = _agent_with_index(index)
    ce = CrossEncoderReranker(model=None)  # only for build_query / build_doc

    records: list[dict] = []
    group = 0
    for sidx, sample in enumerate(samples):
        sid = f"cv_{sidx}"
        agent.reset(sid, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        eff = {**sample, "intent_card": card, "behavior": behavior}
        disclosed, boundary_used = set(), False
        override_applied = sample["scenario_type"] != "intent_override"
        msg = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)

        for turn in range(1, MAX_TURNS + 1):
            resp = agent.respond(sid, msg, turn, TOP_K)
            ctx = agent.retriever.last_context
            pool = agent.retriever.last_candidates
            if ctx is not None and any(a == target for a, _ in pool):
                query = ce.build_query(ctx)
                for asin, feat in pool:
                    records.append({
                        "group": group, "session_idx": sidx, "scenario": sample["scenario_type"],
                        "query": query, "doc": ce.build_doc(index.documents[asin]),
                        "label": 1 if asin == target else 0, "fused": feat[_FUSED],
                    })
            group += 1

            ranked = normalize_recommendations(resp.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            ov = eff.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(ov.get("turn", 3)):
                override_applied = True
                if str(ov.get("new_value", "")):
                    disclosed.add(str(ov["new_value"]))
                msg = str(ov.get("message", "Actually, please ignore my earlier preference."))
            else:
                msg, boundary_used = customer_reply(eff, resp.get("ask_attribute"), disclosed, boundary_used)
    return records


def sample_pairs(records, session_mask, n_hard=8, n_rand=2, seed=0):
    rng = random.Random(seed)
    by_group: dict[int, list[dict]] = {}
    for r in records:
        if session_mask[r["session_idx"]]:
            by_group.setdefault(r["group"], []).append(r)
    out = []
    for rows in by_group.values():
        pos = [r for r in rows if r["label"] == 1]
        if not pos:
            continue
        out.extend(pos)
        neg = sorted((r for r in rows if r["label"] == 0), key=lambda r: -r["fused"])
        out.extend(neg[:n_hard])
        rest = neg[n_hard:]
        out.extend(rng.sample(rest, min(n_rand, len(rest))))
    return out


# --------------------------------------------------------------------------
def fit_linear(data, train_rows, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = [FEATURE_NAMES.index(f) for f in G11]
    X = data.X[np.ix_(train_rows, cols)]
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, random_state=seed).fit(
        sc.transform(X), data.y[train_rows])

    class _Lin:
        trainable = False
        def score(self, rows):
            rows = np.asarray(rows, dtype=np.float64)
            return m.decision_function(sc.transform(rows[:, cols])) if rows.size else np.zeros(0)
    return _Lin()


def finetune(pairs, epochs, seed, model_name=DEFAULT_MODEL, max_length=192, batch_size=16, lr=2e-5):
    import os
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    torch.set_num_threads(os.cpu_count() or 8)
    torch.manual_seed(seed)
    random.Random(seed).shuffle(pairs)
    examples = [InputExample(texts=[p["query"], p["doc"]], label=float(p["label"])) for p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model = CrossEncoder(model_name, num_labels=1, max_length=max_length)
    model.fit(train_dataloader=loader, epochs=epochs,
              warmup_steps=int(0.1 * len(loader) * epochs),
              optimizer_params={"lr": lr}, show_progress_bar=False, use_amp=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=0, help="stop after N folds (0 = all)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--weights", default="0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedGroupKFold

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    cids, cats, prods = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index)
    records = build_text_pairs(args.catalog, args.dataset, index)
    print(f"text pairs: {len(records)}  positives: {sum(r['label'] for r in records)}", flush=True)

    n = int(data.session_idx.max()) + 1
    scen = np.empty(n, dtype=object)
    for i, s in zip(data.session_idx, data.scenario):
        scen[i] = s
    weights = [float(w) for w in args.weights.split(",")]
    arms = ["lr"] + [f"lr+ftce@{w}" for w in weights]
    res = {a: {"hit": [], "mrr": [], "mttc": []} for a in arms}

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr, te) in enumerate(splitter.split(np.arange(n), scen, groups=np.arange(n))):
        if args.max_folds and fold >= args.max_folds:
            break
        mask = np.zeros(n, dtype=bool); mask[tr] = True
        lin = fit_linear(data, sample_training_rows(data, mask, seed=args.seed), args.seed)
        ts = [samples[i] for i in te]

        import time as _t
        _s = _t.time()
        pairs = sample_pairs(records, mask, seed=args.seed)
        ce_model = finetune(pairs, args.epochs, args.seed, args.model, args.max_length)
        print(f"  fold {fold}: {len(pairs)} pairs, fine-tuned in {_t.time() - _s:.0f}s", flush=True)

        for arm in arms:
            agent = _agent_with_index(index)
            agent.retriever.capture = False
            agent.retriever.reranker = lin
            if arm != "lr":
                agent.retriever.cross_encoder = CrossEncoderReranker(
                    ce_model, depth=20, weight=float(arm.split("@")[1]))
            r = evaluate(agent, ts, cids, cats, prods)
            res[arm]["hit"].append(r["hit_rate_at_10"])
            res[arm]["mrr"].append(r["mrr"])
            res[arm]["mttc"].append(r["mttc"])
            print(f"  fold {fold} {arm:12s} hit={r['hit_rate_at_10']:.4f} "
                  f"mrr={r['mrr']:.4f} mttc={r['mttc']:.3f}", flush=True)

    print("\n== CV (real evaluator), fine-tuned cross-encoder ==")
    for arm in arms:
        h = np.asarray(res[arm]["hit"]); m = np.asarray(res[arm]["mrr"]); t = np.asarray(res[arm]["mttc"])
        print(f"  {arm:12s} Hit@10 {h.mean():.4f} ± {h.std():.4f}   "
              f"MRR {m.mean():.4f} ± {m.std():.4f}   MTTC {t.mean():.3f}")


if __name__ == "__main__":
    main()
