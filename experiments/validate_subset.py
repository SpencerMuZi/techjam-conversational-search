"""Real-evaluator 5-fold CV for logistic reranker feature subsets.

Confirms the shrink from feature_select.py does not cost ranking quality once
the actual dialogue loop (early-stop on a hit, clarification cutoff) is back in
play. Reports Hit@10 / MRR / MTTC (mean +/- std over folds) per subset.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.cv import sample_training_rows
from experiments.dataset import _agent_with_index, build_dataset
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.retrieval import CatalogIndex

_G12 = ["log_rating_number", "category_overlap_frac", "constraint_recip", "hard_miss_count",
        "has_price", "profile_hits", "exact_phrase_hits", "soft_idf_sum", "soft_total",
        "hard_total", "soft_hits", "cat_present"]

SUBSETS = {
    "full_36": list(FEATURE_NAMES),
    "g12": _G12,
    # drop soft_idf_sum (sign-unstable across folds)
    "g11": [f for f in _G12 if f != "soft_idf_sum"],
    # also drop soft_total / hard_total (constant within a turn -> no ranking effect)
    "g9": [f for f in _G12 if f not in ("soft_idf_sum", "soft_total", "hard_total")],
}


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
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    cids, cats, prods = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index)
    n_sessions = int(data.session_idx.max()) + 1

    scen = np.empty(n_sessions, dtype=object)
    for i, s in zip(data.session_idx, data.scenario):
        scen[i] = s
    sids = np.arange(n_sessions)
    splitter = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(sids, scen, groups=sids))

    results = {name: {"hit": [], "mrr": [], "mttc": []} for name in SUBSETS}
    for fold, (train_sess, test_sess) in enumerate(folds):
        tr_mask = np.zeros(n_sessions, dtype=bool)
        tr_mask[train_sess] = True
        train_rows = sample_training_rows(data, tr_mask, seed=args.seed)
        test_samples = [samples[i] for i in test_sess]
        for name, feats in SUBSETS.items():
            cols = [FEATURE_NAMES.index(f) for f in feats]
            reranker = _fit(data, train_rows, cols, args.seed)
            agent = _agent_with_index(index)
            agent.retriever.capture = False
            agent.retriever.reranker = reranker
            res = evaluate(agent, test_samples, cids, cats, prods)
            results[name]["hit"].append(res["hit_rate_at_10"])
            results[name]["mrr"].append(res["mrr"])
            results[name]["mttc"].append(res["mttc"])
            print(f"  fold {fold} {name:11s} hit={res['hit_rate_at_10']:.4f} "
                  f"mrr={res['mrr']:.4f} mttc={res['mttc']:.3f}", flush=True)

    print("\n== 5-fold CV (real evaluator) ==")
    print(f"{'subset':12s} {'n':>3s}  {'Hit@10':>16s}  {'MRR':>16s}  {'MTTC':>14s}")
    for name, feats in SUBSETS.items():
        r = results[name]
        def ms(key):
            a = np.asarray(r[key]); return f"{a.mean():.4f} ± {a.std():.4f}"
        print(f"{name:12s} {len(feats):3d}  {ms('hit'):>16s}  {ms('mrr'):>16s}  {ms('mttc'):>14s}")


if __name__ == "__main__":
    main()
