"""Train the shipped logistic-regression reranker on all 200 public sessions.

Writes ``shopping_copilot/reranker_lr.json`` – a ~2 KB file with the standardizer
statistics and linear coefficients. Inference then needs only the standard
library (see ``PackagedLogisticReranker``); scikit-learn is a training-time
dependency only.

    python -m pip install -r requirements-experiments.txt
    python -m experiments.train_reranker
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cv import sample_training_rows
from experiments.dataset import build_dataset
from shopping_copilot.features import FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--out", default="shopping_copilot/reranker_lr.json")
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", default="",
                        help="comma-separated subset of FEATURE_NAMES (default: all 36)")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    selected = [f.strip() for f in args.features.split(",") if f.strip()] or list(FEATURE_NAMES)
    cols = [FEATURE_NAMES.index(name) for name in selected]

    data = build_dataset(args.catalog, args.dataset)
    rows = sample_training_rows(data, np.ones(int(data.session_idx.max()) + 1, dtype=bool), seed=args.seed)
    X, y = data.X[rows][:, cols], data.y[rows]

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=args.C, class_weight="balanced", max_iter=5000, random_state=args.seed
    ).fit(scaler.transform(X), y)

    payload = {
        "model": "logistic_regression",
        "trained_on": "data/public_set.jsonl (200 sessions, full fit)",
        "C": args.C,
        "feature_names": selected,
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "n_rows": int(len(y)),
        "n_positive": int(y.sum()),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  ({len(y)} rows, {int(y.sum())} positives, {len(selected)} features)")

    order = sorted(zip(selected, model.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("standardized coefficients:")
    for name, weight in order:
        print(f"  {name:22s} {weight:+.3f}")


if __name__ == "__main__":
    main()
