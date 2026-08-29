"""A/B: does adding catalog-mined confusables to the training negatives help?

Baseline negatives are the 24 hardest-by-fused-score items already in the
retrieved pool + 8 random. The variant additionally mines, per target, the
`mine_k` catalog items whose own text most resembles the target's (BM25 on the
target's searchable_text) and are NOT in the pool, labels them 0, and trains on
them too.

Both arms use the shipped g11 feature set and are scored by the real evaluator
under the same 5-fold scenario-stratified CV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.retrieval import CatalogIndex

G11 = json.loads(Path("shopping_copilot/reranker_lr.json").read_text(encoding="utf-8"))["feature_names"]


class _SubsetLinear:
    trainable = False

    def __init__(self, cols, scaler, model):
        self._cols = np.asarray(cols)
        self._scaler = scaler
        self._model = model

    def score(self, feature_rows):
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.size == 0:
            return np.zeros((0,))
        return self._model.decision_function(self._scaler.transform(rows[:, self._cols]))


def _fit(data, train_rows, cols, seed):
    X = data.X[np.ix_(train_rows, cols)]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=5000, random_state=seed
    ).fit(scaler.transform(X), data.y[train_rows])
    return _SubsetLinear(cols, scaler, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--mine-k", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    cids, cats, prods = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index, mine_k=args.mine_k)
    cols = [FEATURE_NAMES.index(f) for f in G11]

    n_mined = int(data.is_mined.sum())
    n_pos = int(data.y.sum())
    print(f"mined confusables: {n_mined}  ({n_mined / n_pos:.1f} per positive)")
    # sanity: mined rows should have real constraint/category features, ~0 retrieval
    mrows = data.X[data.is_mined == 1]
    for f in ("category_overlap_frac", "exact_phrase_hits", "constraint_recip", "rrf_fused_score"):
        j = FEATURE_NAMES.index(f)
        print(f"  mined {f:22s} mean={mrows[:, j].mean():.3f}")

    n_sessions = int(data.session_idx.max()) + 1
    scen = np.empty(n_sessions, dtype=object)
    for i, s in zip(data.session_idx, data.scenario):
        scen[i] = s
    sids = np.arange(n_sessions)
    splitter = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    arms = {"baseline": False, f"+mined_{args.mine_k}": True}
    results = {a: {"hit": [], "mrr": [], "mttc": []} for a in arms}
    for fold, (train_sess, test_sess) in enumerate(splitter.split(sids, scen, groups=sids)):
        tr_mask = np.zeros(n_sessions, dtype=bool)
        tr_mask[train_sess] = True
        test_samples = [samples[i] for i in test_sess]
        for arm, use_mined in arms.items():
            rows = sample_training_rows(data, tr_mask, seed=args.seed, include_mined=use_mined)
            reranker = _fit(data, rows, cols, args.seed)
            agent = _agent_with_index(index)
            agent.retriever.capture = False
            agent.retriever.reranker = reranker
            res = evaluate(agent, test_samples, cids, cats, prods)
            results[arm]["hit"].append(res["hit_rate_at_10"])
            results[arm]["mrr"].append(res["mrr"])
            results[arm]["mttc"].append(res["mttc"])
            print(f"  fold {fold} {arm:12s} n_train={len(rows):5d} "
                  f"hit={res['hit_rate_at_10']:.4f} mrr={res['mrr']:.4f} mttc={res['mttc']:.3f}", flush=True)

    print("\n== 5-fold CV (real evaluator), g11 features ==")
    for arm in arms:
        r = results[arm]
        def ms(k):
            a = np.asarray(r[k]); return f"{a.mean():.4f} ± {a.std():.4f}"
        print(f"  {arm:14s} Hit@10 {ms('hit'):>16s}   MRR {ms('mrr'):>16s}   MTTC {ms('mttc'):>14s}")


if __name__ == "__main__":
    main()
