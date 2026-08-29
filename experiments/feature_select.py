"""Shrink the 36-feature reranker to a small, stable subset.

Two selectors, both under 5-fold session-grouped CV with a fast turn-level MRR
proxy (rank the candidates in every held-out turn, take 1/rank of the target):

* L1 path  - LogisticRegression(penalty="l1") over a grid of C; report the order
             in which features enter as the penalty relaxes.
* greedy   - forward selection: repeatedly add the feature that lifts CV MRR most.

The winning subset is then validated with the real evaluator by
``scripts/compare_models.py --features ...`` and ``experiments/train_reranker.py``.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.cv import sample_training_rows
from experiments.dataset import build_dataset
from shopping_copilot.features import FEATURE_NAMES

N_SPLITS = 5
SEED = 0


def _session_scenario(data) -> np.ndarray:
    n = int(data.session_idx.max()) + 1
    out = np.empty(n, dtype=object)
    for idx, scen in zip(data.session_idx, data.scenario):
        out[idx] = scen
    return out


def turn_level_mrr(model, scaler, X_eval, y_eval, groups_eval) -> float:
    """Mean 1/rank of the target across held-out turns that contain the target."""
    scores = model.decision_function(scaler.transform(X_eval))
    rr: list[float] = []
    for g in np.unique(groups_eval):
        mask = groups_eval == g
        if y_eval[mask].sum() == 0:
            continue
        order = np.argsort(-scores[mask], kind="stable")
        target_pos = np.flatnonzero(y_eval[mask][order] == 1)[0] + 1
        rr.append(1.0 / target_pos)
    return float(np.mean(rr)) if rr else 0.0


_FOLDS: list[tuple[np.ndarray, np.ndarray]] = []


def _prepare_folds(data) -> None:
    scen = _session_scenario(data)
    n_sessions = len(scen)
    sids = np.arange(n_sessions)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for train_sess, _ in splitter.split(sids, scen, groups=sids):
        tr_mask = np.zeros(n_sessions, dtype=bool)
        tr_mask[train_sess] = True
        tr_rows = sample_training_rows(data, tr_mask, seed=SEED)
        te_rows = sample_training_rows(data, ~tr_mask, n_hard_neg=10**9, n_rand_neg=0, seed=SEED)
        _FOLDS.append((tr_rows, te_rows))


def cv_mrr(data, feature_cols: list[int]) -> tuple[float, float]:
    cols = np.asarray(feature_cols)
    fold_scores: list[float] = []
    for tr_rows, te_rows in _FOLDS:
        Xtr = data.X[np.ix_(tr_rows, cols)]
        Xte = data.X[np.ix_(te_rows, cols)]
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000, random_state=SEED
        ).fit(scaler.transform(Xtr), data.y[tr_rows])
        fold_scores.append(
            turn_level_mrr(model, scaler, Xte, data.y[te_rows], data.group_id[te_rows])
        )
    arr = np.asarray(fold_scores)
    return float(arr.mean()), float(arr.std())


def l1_path(data) -> list[str]:
    scen = _session_scenario(data)
    rows = sample_training_rows(data, np.ones(len(scen), dtype=bool), seed=SEED)
    X = StandardScaler().fit_transform(data.X[rows])
    y = data.y[rows]
    entered: list[str] = []
    for C in (0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0):
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, class_weight="balanced",
            max_iter=5000, random_state=SEED,
        ).fit(X, y)
        active = [FEATURE_NAMES[i] for i, w in enumerate(model.coef_[0]) if abs(w) > 1e-8]
        newly = [n for n in active if n not in entered]
        entered.extend(newly)
        print(f"  C={C:<6} active={len(active):2d}  +{newly}", flush=True)
    return entered


def greedy(data, max_features: int) -> list[tuple[list[str], float, float]]:
    remaining = list(range(len(FEATURE_NAMES)))
    chosen: list[int] = []
    history: list[tuple[list[str], float, float]] = []
    while remaining and len(chosen) < max_features:
        best = None
        for cand in remaining:
            mean, std = cv_mrr(data, chosen + [cand])
            if best is None or mean > best[1]:
                best = (cand, mean, std)
        cand, mean, std = best
        chosen.append(cand)
        remaining.remove(cand)
        names = [FEATURE_NAMES[i] for i in chosen]
        history.append((list(names), mean, std))
        print(f"  +{FEATURE_NAMES[cand]:24s} ({len(chosen):2d})  CV MRR {mean:.4f} ± {std:.4f}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--max-features", type=int, default=16)
    args = parser.parse_args()

    data = build_dataset(args.catalog, args.dataset)
    _prepare_folds(data)

    full_mean, full_std = cv_mrr(data, list(range(len(FEATURE_NAMES))))
    print(f"\nAll 36 features: CV turn-MRR {full_mean:.4f} ± {full_std:.4f}\n")

    print("L1 path (feature entry order as penalty relaxes):")
    order = l1_path(data)
    print("  ->", order, "\n")

    print("Greedy forward selection:")
    history = greedy(data, args.max_features)

    print("\nCV turn-MRR vs subset size:")
    for names, mean, std in history:
        flag = "  <= within 0.005 of full" if mean >= full_mean - 0.005 else ""
        print(f"  {len(names):2d}  {mean:.4f} ± {std:.4f}{flag}")
    for names, mean, std in history:
        if mean >= full_mean - 0.005:
            print(f"\nsmallest subset within 0.005 of full ({len(names)} features):")
            print("  " + ",".join(names))
            break


if __name__ == "__main__":
    main()
