"""Coefficient stability of the shipped logistic reranker.

Refits the model on every (seed x stratified-by-scenario 5-fold) training split
and reports, per feature: mean / std / |CV| of the standardised coefficient, the
fraction of fits that keep the sign, and the min/max. A feature whose sign flips
across folds, or whose |CV| is large, is fitting noise and should be dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.cv import sample_training_rows
from experiments.dataset import build_dataset
from shopping_copilot.features import FEATURE_NAMES

WEIGHTS_PATH = Path("shopping_copilot/reranker_lr.json")
SEEDS = (0, 1, 2, 3, 4)


def _session_scenario(data) -> np.ndarray:
    n = int(data.session_idx.max()) + 1
    out = np.empty(n, dtype=object)
    for i, s in zip(data.session_idx, data.scenario):
        out[i] = s
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--features", default="",
                        help="comma-separated; default = whatever reranker_lr.json ships")
    parser.add_argument("--C", type=float, default=0.5)
    args = parser.parse_args()

    if args.features.strip():
        feats = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        feats = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))["feature_names"]
    cols = [FEATURE_NAMES.index(f) for f in feats]

    data = build_dataset(args.catalog, args.dataset)
    scen = _session_scenario(data)
    n_sessions = len(scen)
    sids = np.arange(n_sessions)

    coef_runs: list[np.ndarray] = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_sess, _ in splitter.split(sids, scen, groups=sids):
            mask = np.zeros(n_sessions, dtype=bool)
            mask[train_sess] = True
            rows = sample_training_rows(data, mask, seed=seed)
            X = data.X[np.ix_(rows, cols)]
            scaler = StandardScaler().fit(X)
            model = LogisticRegression(
                C=args.C, class_weight="balanced", max_iter=5000, random_state=seed
            ).fit(scaler.transform(X), data.y[rows])
            coef_runs.append(model.coef_[0])
    C = np.vstack(coef_runs)  # (n_fits, n_features)

    # Full-fit reference
    rows = sample_training_rows(data, np.ones(n_sessions, dtype=bool), seed=0)
    Xf = data.X[np.ix_(rows, cols)]
    full = LogisticRegression(
        C=args.C, class_weight="balanced", max_iter=5000, random_state=0
    ).fit(StandardScaler().fit_transform(Xf), data.y[rows]).coef_[0]

    print(f"\n{len(coef_runs)} fits ({len(SEEDS)} seeds x 5 folds), {len(feats)} features, C={args.C}\n")
    print(f"{'feature':22s} {'full':>8s} {'mean':>8s} {'std':>7s} {'|CV|':>6s} {'sign%':>6s} "
          f"{'min':>8s} {'max':>8s}  flag")
    order = np.argsort(-np.abs(full))
    for j in order:
        col = C[:, j]
        mean, std = col.mean(), col.std()
        cv = abs(std / mean) if abs(mean) > 1e-9 else float("inf")
        sign_frac = np.mean(np.sign(col) == np.sign(mean)) if abs(mean) > 1e-9 else 0.0
        flag = ""
        if sign_frac < 0.9:
            flag = "SIGN UNSTABLE"
        elif cv > 0.5:
            flag = "high variance"
        print(f"{feats[j]:22s} {full[j]:+8.3f} {mean:+8.3f} {std:7.3f} {cv:6.2f} "
              f"{sign_frac*100:5.0f}% {col.min():+8.3f} {col.max():+8.3f}  {flag}")

    ranking_effective = [f for f in feats if f not in ("soft_total", "hard_total", "turn", "route_is_buying")]
    print(f"\nranking-effective features (exclude group-constants): {len(ranking_effective)}")


if __name__ == "__main__":
    main()
