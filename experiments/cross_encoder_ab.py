"""A/B: does a local cross-encoder on top of the g11 logistic reranker help?

Real-evaluator 5-fold scenario-stratified CV. The logistic model is refit per
fold; the cross-encoder is frozen (pretrained MS-MARCO MiniLM). Compares the
linear reranker alone against linear + cross-encoder at a few blend weights.

    python -m pip install -r requirements-experiments.txt
    python -m experiments.cross_encoder_ab --weights 0.5,0.7,0.9
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
from shopping_copilot.cross_encoder import load_cross_encoder
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
    parser.add_argument("--weights", default="0.5,0.7,0.9")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ce = load_cross_encoder(True, depth=args.depth, weight=0.7)
    if ce is None:
        raise SystemExit("cross-encoder unavailable: pip install sentence-transformers and retry")

    index = CatalogIndex(args.catalog)
    samples = load_jsonl(args.dataset)
    cids, cats, prods = catalog_index(args.catalog)
    data = build_dataset(args.catalog, args.dataset, index=index)
    cols = [FEATURE_NAMES.index(f) for f in G11]
    weights = [float(w) for w in args.weights.split(",")]

    n_sessions = int(data.session_idx.max()) + 1
    scen = np.empty(n_sessions, dtype=object)
    for i, s in zip(data.session_idx, data.scenario):
        scen[i] = s
    sids = np.arange(n_sessions)
    splitter = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    arms = ["lr"] + [f"lr+ce@{w}" for w in weights]
    results = {a: {"hit": [], "mrr": [], "mttc": []} for a in arms}

    for fold, (train_sess, test_sess) in enumerate(splitter.split(sids, scen, groups=sids)):
        tr_mask = np.zeros(n_sessions, dtype=bool)
        tr_mask[train_sess] = True
        train_rows = sample_training_rows(data, tr_mask, seed=args.seed)
        lin = _fit(data, train_rows, cols, args.seed)
        test_samples = [samples[i] for i in test_sess]

        for arm in arms:
            agent = _agent_with_index(index)
            agent.retriever.capture = False
            agent.retriever.reranker = lin
            if arm != "lr":
                ce.weight = float(arm.split("@")[1])
                agent.retriever.cross_encoder = ce
            res = evaluate(agent, test_samples, cids, cats, prods)
            results[arm]["hit"].append(res["hit_rate_at_10"])
            results[arm]["mrr"].append(res["mrr"])
            results[arm]["mttc"].append(res["mttc"])
            print(f"  fold {fold} {arm:12s} hit={res['hit_rate_at_10']:.4f} "
                  f"mrr={res['mrr']:.4f} mttc={res['mttc']:.3f}", flush=True)

    print("\n== 5-fold CV (real evaluator), g11 features ==")
    for arm in arms:
        r = results[arm]
        def ms(k):
            a = np.asarray(r[k]); return f"{a.mean():.4f} ± {a.std():.4f}"
        print(f"  {arm:12s} Hit@10 {ms('hit'):>16s}   MRR {ms('mrr'):>16s}   MTTC {ms('mttc'):>14s}")


if __name__ == "__main__":
    main()
